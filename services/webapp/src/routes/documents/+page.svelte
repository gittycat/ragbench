<script lang="ts">
	import { SvelteSet } from 'svelte/reactivity';
	import { onMount } from 'svelte';
	import {
		fetchDocuments,
		deleteDocument,
		type Document,
		type DocumentSortField,
		type SortOrder
	} from '$lib/api';

	let documents = $state<Document[]>([]);
	let selectedIds = new SvelteSet<string>();
	let isLoading = $state(true);
	let error = $state<string | null>(null);
	let isDeleting = $state(false);

	// Sorting state
	let sortBy = $state<DocumentSortField>('uploaded_at');
	let sortOrder = $state<SortOrder>('desc');

	// Pagination - limit visible rows
	const MAX_VISIBLE_ROWS = 15;

	const allSelected = $derived(documents.length > 0 && selectedIds.size === documents.length);
	const someSelected = $derived(selectedIds.size > 0);

	// Calculate visible documents and remaining count
	const visibleDocuments = $derived(documents.slice(0, MAX_VISIBLE_ROWS));
	const remainingCount = $derived(Math.max(0, documents.length - MAX_VISIBLE_ROWS));

	onMount(async () => {
		await loadDocuments();
	});

	async function loadDocuments() {
		isLoading = true;
		error = null;
		try {
			documents = await fetchDocuments(sortBy, sortOrder);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load documents';
		} finally {
			isLoading = false;
		}
	}

	function formatUploadTime(isoString: string | undefined): string {
		if (!isoString) return '—';
		try {
			const date = new Date(isoString);
			const seconds = Math.floor((Date.now() - date.getTime()) / 1000);

			if (seconds < 60) return 'just now';

			const minutes = Math.floor(seconds / 60);
			if (minutes < 60) return `${minutes} min ago`;

			const hours = Math.floor(minutes / 60);
			if (hours < 24) return `${hours} hour${hours === 1 ? '' : 's'} ago`;

			const days = Math.floor(hours / 24);
			if (days < 7) return `${days} day${days === 1 ? '' : 's'} ago`;

			return date.toISOString().split('T')[0]; // YYYY-MM-DD
		} catch {
			return '—';
		}
	}

	function toggleSelectAll() {
		if (allSelected) {
			selectedIds.clear();
		} else {
			documents.forEach((d) => selectedIds.add(d.id));
		}
	}

	function toggleSelect(id: string) {
		if (selectedIds.has(id)) {
			selectedIds.delete(id);
		} else {
			selectedIds.add(id);
		}
	}

	async function handleDeleteDocument(id: string) {
		if (!confirm('Are you sure you want to delete this document?')) return;

		isDeleting = true;
		try {
			await deleteDocument(id);
			selectedIds.delete(id);
			await loadDocuments();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to delete document';
		} finally {
			isDeleting = false;
		}
	}

	async function handleDeleteSelected() {
		if (!confirm(`Are you sure you want to delete ${selectedIds.size} document(s)?`)) return;

		isDeleting = true;
		try {
			const idsToDelete = Array.from(selectedIds);
			for (const id of idsToDelete) {
				await deleteDocument(id);
			}
			selectedIds.clear();
			await loadDocuments();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to delete documents';
		} finally {
			isDeleting = false;
		}
	}

	async function handleSort(field: DocumentSortField) {
		if (sortBy === field) {
			// Toggle order if same field
			sortOrder = sortOrder === 'asc' ? 'desc' : 'asc';
		} else {
			// New field, default to desc for uploaded_at and chunks, asc for name
			sortBy = field;
			sortOrder = field === 'name' ? 'asc' : 'desc';
		}
		await loadDocuments();
	}

	function getSortIcon(field: DocumentSortField): string {
		if (sortBy !== field) return '⇅';
		return sortOrder === 'asc' ? '↑' : '↓';
	}
</script>

<div class="flex flex-col h-full gap-4">
	<!-- Action Bar -->
	<div class="flex items-center gap-2 bg-base-200 px-3 py-2 rounded-lg">
		<div class="tooltip tooltip-bottom" data-tip="Delete selected ({selectedIds.size})">
			<button
				class="btn btn-sm btn-square btn-action text-error disabled:text-base-content/30"
				disabled={!someSelected || isDeleting}
				onclick={handleDeleteSelected}
				aria-label="Delete selected documents"
			>
			{#if isDeleting}
				<span class="loading loading-spinner loading-xs"></span>
			{:else}
				<svg
					xmlns="http://www.w3.org/2000/svg"
					class="h-4 w-4"
					fill="none"
					viewBox="0 0 24 24"
					stroke="currentColor"
				>
					<path
						stroke-linecap="round"
						stroke-linejoin="round"
						stroke-width="2"
						d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
					/>
				</svg>
			{/if}
			</button>
		</div>
		<div class="tooltip tooltip-bottom" data-tip="Refresh">
			<button
				class="btn btn-sm btn-square btn-action"
				onclick={loadDocuments}
				disabled={isLoading}
				aria-label="Refresh document list"
			>
			{#if isLoading}
				<span class="loading loading-spinner loading-xs"></span>
			{:else}
				<svg
					xmlns="http://www.w3.org/2000/svg"
					class="h-4 w-4"
					fill="none"
					viewBox="0 0 24 24"
					stroke="currentColor"
				>
					<path
						stroke-linecap="round"
						stroke-linejoin="round"
						stroke-width="2"
						d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
					/>
				</svg>
			{/if}
			</button>
		</div>
	</div>

	<!-- Error Alert -->
	{#if error}
		<div class="alert alert-error">
			<svg
				xmlns="http://www.w3.org/2000/svg"
				class="h-6 w-6 shrink-0"
				fill="none"
				viewBox="0 0 24 24"
				stroke="currentColor"
			>
				<path
					stroke-linecap="round"
					stroke-linejoin="round"
					stroke-width="2"
					d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
				/>
			</svg>
			<span>{error}</span>
			<button class="btn btn-ghost btn-sm" onclick={() => (error = null)}>Dismiss</button>
		</div>
	{/if}

	<!-- Documents Table -->
	<div class="overflow-x-auto flex-1">
		{#if isLoading && documents.length === 0}
			<div class="flex items-center justify-center h-full">
				<span class="loading loading-spinner loading-lg"></span>
			</div>
		{:else}
			<table class="table table-xs table-pin-rows">
				<thead>
					<tr class="bg-base-200">
						<th class="w-8">
							<label>
								<input
									type="checkbox"
									class="checkbox checkbox-xs"
									checked={allSelected}
									onchange={toggleSelectAll}
								/>
							</label>
						</th>
						<th>
							<button
								class="flex items-center gap-1 hover:text-primary transition-colors"
								onclick={() => handleSort('name')}
								title="Sort by name"
							>
								Name
								<span class="text-xs opacity-60">{getSortIcon('name')}</span>
							</button>
						</th>
						<th class="w-24">
							<div class="tooltip tooltip-bottom" data-tip="Text segments created for search indexing">
								<button
									class="flex items-center gap-1 w-full justify-end hover:text-primary transition-colors"
									onclick={() => handleSort('chunks')}
								>
									Chunks
									<span class="text-xs opacity-60">{getSortIcon('chunks')}</span>
								</button>
							</div>
						</th>
						<th class="w-44">
							<button
								class="flex items-center gap-1 w-full justify-end hover:text-primary transition-colors"
								onclick={() => handleSort('uploaded_at')}
								title="Sort by upload time"
							>
								Added
								<span class="text-xs opacity-60">{getSortIcon('uploaded_at')}</span>
							</button>
						</th>
						<th class="w-16"></th>
					</tr>
				</thead>
				<tbody>
					{#each visibleDocuments as doc (doc.id)}
						<tr class="hover">
							<th>
								<label>
									<input
										type="checkbox"
										class="checkbox checkbox-xs"
										checked={selectedIds.has(doc.id)}
										onchange={() => toggleSelect(doc.id)}
									/>
								</label>
							</th>
							<td class="font-mono text-xs truncate max-w-md" title={doc.file_name}>
								{doc.file_name}
							</td>
							<td class="text-right text-xs">{doc.chunks}</td>
							<td class="text-right text-xs font-mono text-base-content/70">
								{formatUploadTime(doc.uploaded_at)}
							</td>
							<td>
								<button
									class="btn btn-ghost btn-xs text-error"
									onclick={() => handleDeleteDocument(doc.id)}
									title="Delete document"
									aria-label="Delete {doc.file_name}"
									disabled={isDeleting}
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										class="h-4 w-4"
										fill="none"
										viewBox="0 0 24 24"
										stroke="currentColor"
									>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											stroke-width="2"
											d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
										/>
									</svg>
								</button>
							</td>
						</tr>
					{:else}
						<tr>
							<td colspan="5" class="text-center py-8 text-base-content/50">
								No documents indexed yet.
								<a href="/upload" class="link link-primary">Upload documents</a> to get started.
							</td>
						</tr>
					{/each}
				</tbody>
			</table>

			<!-- Remaining documents indicator -->
			{#if remainingCount > 0}
				<div class="text-center py-3 text-sm text-base-content/60 bg-base-200/50 border-t border-base-300">
					+{remainingCount} more document{remainingCount === 1 ? '' : 's'} stored
				</div>
			{/if}
		{/if}
	</div>
</div>
