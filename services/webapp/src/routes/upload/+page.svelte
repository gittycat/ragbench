<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import { onMount, tick } from 'svelte';
	import {
		uploadFiles,
		fetchBatchProgress,
		computeFileHash,
		checkDuplicateFiles,
		type BatchProgressResponse,
		type TaskStatus,
		type FileCheckItem
	} from '$lib/api';

	interface UploadItem {
		id: string;
		filename: string;
		size: number;
		uploadProgress: number;
		processingProgress: number;
		status: 'uploading' | 'processing' | 'done' | 'error' | 'skipped';
		error?: string;
		skipReason?: string;
		taskId?: string;
		batchId?: string;
	}

	// Weight constants for combined progress
	const UPLOAD_WEIGHT = 0.1; // 10% for upload
	const PROCESSING_WEIGHT = 0.9; // 90% for processing

	function calculateOverallProgress(item: UploadItem): number {
		if (item.status === 'skipped') return 0;
		if (item.status === 'done') return 100;

		// Upload contributes 10%, processing contributes 90%
		const uploadContribution = item.uploadProgress * UPLOAD_WEIGHT;
		const processingContribution = item.processingProgress * PROCESSING_WEIGHT;

		return Math.round(uploadContribution + processingContribution);
	}

	let uploads = $state<UploadItem[]>([]);
	let fileInput: HTMLInputElement;
	let dirInput: HTMLInputElement;
	let isUploading = $state(false);
	let ollamaError = $state<string | null>(null);

	// Active batches being polled
	let activeBatches = $state<Set<string>>(new Set());

	// Auto-trigger file picker based on query parameter
	onMount(async () => {
		await tick();
		const trigger = $page.url.searchParams.get('trigger');

		// Set up cancel handlers to redirect back to Documents if user cancels picker
		const handleCancel = () => {
			// Only redirect if no uploads are in progress
			if (uploads.length === 0) {
				goto('/documents');
			}
		};

		if (trigger === 'files' && fileInput) {
			fileInput.addEventListener('cancel', handleCancel);
			fileInput.click();
		} else if (trigger === 'directory' && dirInput) {
			dirInput.addEventListener('cancel', handleCancel);
			dirInput.click();
		}
	});

	function formatSize(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
	}

	async function handleFileUpload(event: Event) {
		const input = event.target as HTMLInputElement;
		if (!input.files || input.files.length === 0) return;
		await processUpload(Array.from(input.files));
		input.value = '';
	}

	async function handleDirUpload(event: Event) {
		const input = event.target as HTMLInputElement;
		if (!input.files || input.files.length === 0) return;
		await processUpload(Array.from(input.files));
		input.value = '';
	}

	async function processUpload(files: File[]) {
		isUploading = true;

		// Create upload items for UI
		const newItems: UploadItem[] = files.map((file, idx) => ({
			id: `upload-${Date.now()}-${idx}`,
			filename: file.webkitRelativePath || file.name,
			size: file.size,
			uploadProgress: 0,
			processingProgress: 0,
			status: 'uploading' as const
		}));

		uploads = [...newItems, ...uploads];

		try {
			// Step 1: Compute hashes for all files
			const fileChecks: FileCheckItem[] = [];
			const fileMap = new Map<string, File>();

			for (let i = 0; i < files.length; i++) {
				const file = files[i];
				const filename = file.webkitRelativePath || file.name;

				// Show progress for hashing
				uploads = uploads.map((item) => {
					if (item.id === newItems[i].id) {
						return { ...item, uploadProgress: 10 };
					}
					return item;
				});

				const hash = await computeFileHash(file);
				fileChecks.push({ filename, size: file.size, hash });
				fileMap.set(filename, file);

				// Update progress after hashing
				uploads = uploads.map((item) => {
					if (item.id === newItems[i].id) {
						return { ...item, uploadProgress: 30 };
					}
					return item;
				});
			}

			// Step 2: Check for duplicates
			const duplicateCheck = await checkDuplicateFiles(fileChecks);

			// Step 3: Separate files into upload vs skipped
			const filesToUpload: File[] = [];
			const skippedFiles = new Set<string>();

			for (const filename of fileMap.keys()) {
				const checkResult = duplicateCheck.results[filename];
				if (checkResult?.exists) {
					skippedFiles.add(filename);
					// Mark as skipped immediately
					uploads = uploads.map((item) => {
						if (newItems.some((n) => n.id === item.id && n.filename === filename)) {
							return {
								...item,
								status: 'skipped' as const,
								skipReason: checkResult.reason || 'Already uploaded'
							};
						}
						return item;
					});
				} else {
					filesToUpload.push(fileMap.get(filename)!);
				}
			}

			// Step 4: Upload non-duplicate files only
			if (filesToUpload.length === 0) {
				// All files were duplicates
				isUploading = false;
				return;
			}

			// Simulate upload progress for files being uploaded
			const uploadProgressInterval = setInterval(() => {
				uploads = uploads.map((item) => {
					if (
						newItems.some((n) => n.id === item.id) &&
						item.status === 'uploading' &&
						!skippedFiles.has(item.filename)
					) {
						const newProgress = Math.min(item.uploadProgress + 20, 90);
						return { ...item, uploadProgress: newProgress };
					}
					return item;
				});
			}, 100);

			const response = await uploadFiles(filesToUpload);

			clearInterval(uploadProgressInterval);

			// Mark upload as complete, update with task IDs, start processing
			// Track matched tasks to avoid duplicate assignments
			const matchedTaskIds = new Set<string>();

			uploads = uploads.map((item) => {
				if (!newItems.some((n) => n.id === item.id)) {
					return item;
				}

				if (skippedFiles.has(item.filename)) {
					return item;
				}

				// Try exact match first, then fallback to suffix match
				const matchingTask = response.tasks.find(
					(t) => !matchedTaskIds.has(t.task_id) && (
						t.filename === item.filename || item.filename.endsWith(t.filename)
					)
				);

				if (matchingTask) {
					matchedTaskIds.add(matchingTask.task_id);
					return {
						...item,
						uploadProgress: 100,
						status: 'processing' as const,
						taskId: matchingTask.task_id,
						batchId: response.batch_id
					};
				} else {
					// No matching task found - this shouldn't happen but handle gracefully
					console.warn(`No matching task found for file: ${item.filename}`);
					return {
						...item,
						uploadProgress: 100,
						status: 'error' as const,
						error: 'Failed to match upload task'
					};
				}
			});

			// Start polling for this batch
			activeBatches.add(response.batch_id);
			pollBatchProgress(response.batch_id);
		} catch (error) {
			// Show prominent alert for Ollama connectivity issues
			if (error instanceof Error && error.name === 'ServiceUnavailable') {
				ollamaError = error.message;
			}

			// Mark all non-skipped items as error
			uploads = uploads.map((item) => {
				if (newItems.some((n) => n.id === item.id) && item.status !== 'skipped') {
					return {
						...item,
						status: 'error' as const,
						error: error instanceof Error ? error.message : 'Upload failed'
					};
				}
				return item;
			});
		} finally {
			isUploading = false;
		}
	}

	async function pollBatchProgress(batchId: string) {
		const pollInterval = setInterval(async () => {
			try {
				const progress = await fetchBatchProgress(batchId);
				updateProcessingProgress(progress);

				// Check if all tasks are complete
				const allDone = Object.values(progress.tasks).every(
					(t) => t.status === 'completed' || t.status === 'error'
				);

				if (allDone) {
					clearInterval(pollInterval);
					activeBatches.delete(batchId);
				}
			} catch {
				// Batch might have expired or error occurred
				clearInterval(pollInterval);
				activeBatches.delete(batchId);
			}
		}, 1000);
	}

	function updateProcessingProgress(progress: BatchProgressResponse) {
		uploads = uploads.map((item) => {
			if (item.batchId !== progress.batch_id) return item;

			const taskStatus = item.taskId ? progress.tasks[item.taskId] : null;
			if (!taskStatus) return item;

			if (taskStatus.status === 'completed') {
				return { ...item, processingProgress: 100, status: 'done' as const };
			} else if (taskStatus.status === 'error') {
				return {
					...item,
					status: 'error' as const,
					error: taskStatus.data?.error || 'Processing failed'
				};
			} else if (taskStatus.total_chunks && taskStatus.completed_chunks !== undefined) {
				const pct = Math.round((taskStatus.completed_chunks / taskStatus.total_chunks) * 100);
				return { ...item, processingProgress: pct };
			} else if (taskStatus.status === 'processing') {
				// Indeterminate progress
				return { ...item, processingProgress: Math.min(item.processingProgress + 5, 90) };
			}
			return item;
		});
	}

	function clearCompleted() {
		uploads = uploads.filter((u) => u.status !== 'done' && u.status !== 'error' && u.status !== 'skipped');
	}

	function getStatusBadgeClass(status: UploadItem['status']): string {
		switch (status) {
			case 'uploading':
				return 'badge-info';
			case 'processing':
				return 'badge-warning';
			case 'done':
				return 'badge-success';
			case 'error':
				return 'badge-error';
			case 'skipped':
				return 'badge-ghost';
			default:
				return 'badge-ghost';
		}
	}
</script>

<div class="flex flex-col h-full gap-4">
	{#if ollamaError}
		<div role="alert" class="alert alert-error shadow-lg">
			<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 shrink-0 stroke-current" fill="none" viewBox="0 0 24 24">
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
			</svg>
			<div>
				<h3 class="font-bold">Ollama is not accessible</h3>
				<p class="text-sm">Ollama is required for generating embeddings. Check that it is running and reachable, then try uploading again.</p>
			</div>
			<button class="btn btn-sm btn-ghost" onclick={() => ollamaError = null}>Dismiss</button>
		</div>
	{/if}

	<!-- Hidden file inputs (triggered from Documents page) -->
	<input
		bind:this={fileInput}
		type="file"
		multiple
		class="hidden"
		onchange={handleFileUpload}
		accept=".txt,.md,.pdf,.docx,.pptx,.xlsx,.html,.htm,.asciidoc,.adoc"
	/>
	<input
		bind:this={dirInput}
		type="file"
		webkitdirectory
		class="hidden"
		onchange={handleDirUpload}
	/>

	<!-- Action Bar -->
	<div class="flex gap-2 items-center">
		{#if uploads.some((u) => u.status === 'done' || u.status === 'error' || u.status === 'skipped')}
			<button class="btn btn-ghost btn-sm" onclick={clearCompleted}>Clear Completed</button>
		{/if}
		<div class="flex-1"></div>
		<span class="text-sm text-base-content/60">{uploads.length} uploads</span>
	</div>

	<!-- Supported formats info -->
	<div class="text-xs text-base-content/50">
		Supported formats: PDF, DOCX, PPTX, XLSX, HTML, TXT, MD, AsciiDoc
	</div>

	<!-- Upload Progress Table -->
	<div class="overflow-x-auto flex-1">
		<table class="table table-xs table-pin-rows">
			<thead>
				<tr class="bg-base-200">
					<th>Document</th>
					<th class="w-24 text-right">Size</th>
					<th class="w-48">Progress</th>
					<th class="w-24">Status</th>
				</tr>
			</thead>
			<tbody>
				{#each uploads as upload (upload.id)}
					<tr class="hover">
						<td class="font-mono text-xs truncate max-w-md" title={upload.filename}>
							{upload.filename}
						</td>
						<td class="text-right text-xs">{formatSize(upload.size)}</td>
						<td>
							{#if upload.status === 'skipped'}
								<span class="text-xs text-base-content/40">—</span>
							{:else if upload.status === 'error'}
								<span class="text-xs text-error">Failed</span>
							{:else}
								{@const overallProgress = calculateOverallProgress(upload)}
								{@const progressClass = upload.status === 'done' ? 'progress-success' : overallProgress < 10 ? 'progress-info' : 'progress-warning'}
								<div class="flex items-center gap-2">
									<progress
										class="progress {progressClass} w-32"
										value={overallProgress}
										max="100"
									></progress>
									<span class="text-xs text-base-content/60">{overallProgress}%</span>
								</div>
							{/if}
						</td>
						<td class="relative">
							{#if upload.status === 'error' && upload.error}
								<div class="tooltip tooltip-error tooltip-top z-50 before:-translate-y-1" data-tip={upload.error}>
									<span class="badge badge-sm {getStatusBadgeClass(upload.status)}">
										Error
									</span>
								</div>
							{:else if upload.status === 'skipped' && upload.skipReason}
								<div class="tooltip tooltip-info tooltip-top z-50 before:-translate-y-1" data-tip={upload.skipReason}>
									<span class="badge badge-sm {getStatusBadgeClass(upload.status)}">
										Skipped
									</span>
								</div>
							{:else}
								<span class="badge badge-sm {getStatusBadgeClass(upload.status)}">
									{upload.status === 'done'
										? 'Done'
										: upload.status === 'error'
											? 'Error'
											: upload.status === 'uploading'
												? 'Uploading'
												: upload.status === 'skipped'
													? 'Skipped'
													: 'Processing'}
								</span>
							{/if}
						</td>
					</tr>
				{:else}
					<tr>
						<td colspan="4" class="text-center py-8 text-base-content/50">
							No uploads in progress. Use the buttons above to upload documents.
						</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
</div>
