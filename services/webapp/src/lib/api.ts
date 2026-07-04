// RAG Server API Client

const API_BASE = '/api';

// ============================================================================
// Types - Documents
// ============================================================================

export interface Document {
	id: string;
	file_name: string;
	chunks: number;
	file_size_bytes?: number;
	uploaded_at?: string; // ISO 8601 timestamp (e.g., "2025-01-15T14:30:00Z")
}

export interface DocumentListResponse {
	documents: Document[];
}

// ============================================================================
// Types - Upload
// ============================================================================

export interface TaskInfo {
	task_id: string;
	filename: string;
}

export interface BatchUploadResponse {
	status: string;
	batch_id: string;
	tasks: TaskInfo[];
}

export interface TaskStatus {
	status: 'pending' | 'processing' | 'completed' | 'error';
	filename: string;
	progress?: number;
	total_chunks?: number;
	completed_chunks?: number;
	error?: string;
	data?: {
		document_id?: string;
		chunks?: number;
		error?: string;
		message?: string;
	};
}

export interface BatchProgressResponse {
	batch_id: string;
	total: number;
	completed: number;
	tasks: Record<string, TaskStatus>;
}

export interface FileCheckItem {
	filename: string;
	size: number;
	hash: string;
}

export interface FileCheckResult {
	filename: string;
	exists: boolean;
	document_id?: string;
	existing_filename?: string;
	reason?: string;
}

export interface FileCheckResponse {
	results: Record<string, FileCheckResult>;
}

// ============================================================================
// Types - Metrics/Analytics
// ============================================================================

export interface ModelSize {
	parameters?: string;
	disk_size_mb?: number;
	context_window?: number;
}

export interface ModelInfo {
	name: string;
	provider: string;
	model_type: string;
	is_local: boolean;
	size?: ModelSize;
	reference_url?: string;
	description?: string;
	status: string;
}

export interface ModelsConfig {
	llm: ModelInfo;
	embedding: ModelInfo;
	reranker?: ModelInfo;
	eval?: ModelInfo;
}

export interface BM25Config {
	enabled: boolean;
	description: string;
	strengths: string[];
}

export interface VectorSearchConfig {
	enabled: boolean;
	chunk_size: number;
	chunk_overlap: number;
	vector_store: string;
	collection_name: string;
}

export interface HybridSearchConfig {
	enabled: boolean;
	bm25: BM25Config;
	vector: VectorSearchConfig;
	fusion_method: string;
	rrf_k: number;
	description: string;
	research_reference: string;
	improvement_claim: string;
}

export interface ContextualRetrievalConfig {
	enabled: boolean;
	description: string;
	research_reference: string;
	improvement_claim: string;
	performance_impact: string;
}

export interface RerankerConfig {
	enabled: boolean;
	model?: string;
	top_n?: number;
	description: string;
	reference_url?: string;
}

export interface RetrievalConfig {
	retrieval_top_k: number;
	final_top_n: number;
	hybrid_search: HybridSearchConfig;
	contextual_retrieval: ContextualRetrievalConfig;
	reranker: RerankerConfig;
	pipeline_description: string;
}

export interface MetricDefinition {
	name: string;
	category: string;
	description: string;
	threshold: number;
	interpretation: string;
	reference_url?: string;
}

export interface SystemMetrics {
	system_name: string;
	version: string;
	timestamp: string;
	models: ModelsConfig;
	retrieval: RetrievalConfig;
	evaluation_metrics: MetricDefinition[];
	document_count: number;
	chunk_count: number;
	health_status: string;
	component_status: Record<string, string>;
}

// ============================================================================
// API Functions - Documents
// ============================================================================

export type DocumentSortField = 'name' | 'chunks' | 'uploaded_at';
export type SortOrder = 'asc' | 'desc';

export async function fetchDocuments(
	sortBy: DocumentSortField = 'uploaded_at',
	sortOrder: SortOrder = 'desc'
): Promise<Document[]> {
	const params = new URLSearchParams();
	params.set('sort_by', sortBy);
	params.set('sort_order', sortOrder);

	const response = await fetch(`${API_BASE}/documents?${params}`);
	if (!response.ok) {
		throw new Error(`Failed to fetch documents: ${response.statusText}`);
	}
	const data: DocumentListResponse = await response.json();
	return data.documents;
}

export async function deleteDocument(documentId: string): Promise<void> {
	const response = await fetch(`${API_BASE}/documents/${documentId}`, {
		method: 'DELETE'
	});
	if (!response.ok) {
		throw new Error(`Failed to delete document: ${response.statusText}`);
	}
}

// ============================================================================
// API Functions - Upload
// ============================================================================

export async function uploadFiles(files: FileList | File[]): Promise<BatchUploadResponse> {
	const formData = new FormData();
	for (const file of files) {
		// Preserve relative path for directory uploads
		// The third parameter sets the filename that gets sent to the server
		const filename = ('webkitRelativePath' in file && file.webkitRelativePath)
			? file.webkitRelativePath
			: file.name;
		formData.append('files', file, filename);
	}

	const response = await fetch(`${API_BASE}/upload`, {
		method: 'POST',
		body: formData
	});

	if (!response.ok) {
		const errorText = await response.text();
		let detail = errorText;
		try {
			const parsed = JSON.parse(errorText);
			if (parsed.detail) detail = parsed.detail;
		} catch { /* use raw text */ }

		if (response.status === 503) {
			const err = new Error(detail);
			err.name = 'ServiceUnavailable';
			throw err;
		}
		throw new Error(`Upload failed: ${detail}`);
	}

	return response.json();
}

export async function fetchBatchProgress(batchId: string): Promise<BatchProgressResponse> {
	const response = await fetch(`${API_BASE}/tasks/${batchId}/status`);
	if (!response.ok) {
		throw new Error(`Failed to fetch batch progress: ${response.statusText}`);
	}
	return response.json();
}

/**
 * Compute SHA256 hash of a file using Web Crypto API.
 * Matches LlamaIndex's document hashing approach.
 */
export async function computeFileHash(file: File): Promise<string> {
	const buffer = await file.arrayBuffer();
	const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
	const hashArray = Array.from(new Uint8Array(hashBuffer));
	const hashHex = hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
	return hashHex;
}

/**
 * Check if files with given hashes already exist in the system.
 * Returns information about which files are duplicates.
 */
export async function checkDuplicateFiles(
	files: FileCheckItem[]
): Promise<FileCheckResponse> {
	const response = await fetch(`${API_BASE}/documents/check-duplicates`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({ files })
	});

	if (!response.ok) {
		throw new Error(`Failed to check duplicates: ${response.statusText}`);
	}

	return response.json();
}

// ============================================================================
// API Functions - Metrics/Analytics
// ============================================================================

export async function fetchSystemMetrics(): Promise<SystemMetrics> {
	const response = await fetch(`${API_BASE}/metrics/system`);
	if (!response.ok) {
		throw new Error(`Failed to fetch system metrics: ${response.statusText}`);
	}
	return response.json();
}

export async function fetchHealth(): Promise<{ status: string }> {
	const response = await fetch(`${API_BASE}/health`);
	if (!response.ok) {
		throw new Error(`Failed to fetch health: ${response.statusText}`);
	}
	return response.json();
}

// ============================================================================
// Types - Chat
// ============================================================================

export interface ChatSource {
	document_id: string | null;
	document_name: string;
	excerpt: string;
	full_text: string;
	path: string;
	score: number | null;
}

export interface ChatMessage {
	role: 'user' | 'assistant';
	content: string;
	sources?: ChatSource[];
	timestamp?: string;
}

export interface QueryResponse {
	answer: string;
	sources: ChatSource[];
	session_id: string;
}

export interface ChatHistoryMessage {
	role: string;
	content: string;
}

export interface ChatHistoryResponse {
	session_id: string;
	messages: ChatHistoryMessage[];
	metadata?: SessionMetadata;
}

export type SSEEventType = 'token' | 'sources' | 'done' | 'error';

export interface SSETokenEvent {
	event: 'token';
	data: { token: string };
}

export interface SSESourcesEvent {
	event: 'sources';
	data: { sources: ChatSource[]; session_id: string };
}

export interface SSEDoneEvent {
	event: 'done';
	data: Record<string, never>;
}

export interface SSEErrorEvent {
	event: 'error';
	data: { error: string };
}

export type SSEEvent = SSETokenEvent | SSESourcesEvent | SSEDoneEvent | SSEErrorEvent;

// ============================================================================
// API Functions - Chat
// ============================================================================

/**
 * Stream a RAG query response using Server-Sent Events.
 * Yields SSE events as they arrive from the server.
 * @param signal - Optional AbortSignal to cancel the request
 */
export async function* streamQuery(
	query: string,
	sessionId: string | null,
	isTemporary: boolean = false,
	signal?: AbortSignal
): AsyncGenerator<SSEEvent, void, undefined> {
	const response = await fetch(`${API_BASE}/query/stream`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({ query, session_id: sessionId, is_temporary: isTemporary }),
		signal
	});

	if (!response.ok) {
		throw new Error(`Query failed: ${response.statusText}`);
	}

	if (!response.body) {
		throw new Error('No response body');
	}

	const reader = response.body.getReader();
	const decoder = new TextDecoder();
	let buffer = '';

	try {
		while (true) {
			const { done, value } = await reader.read();
			if (done) break;

			buffer += decoder.decode(value, { stream: true });

			// Parse SSE events from buffer
			const lines = buffer.split('\n');
			buffer = lines.pop() || ''; // Keep incomplete line in buffer

			let currentEvent: SSEEventType | null = null;
			let currentData = '';

			for (const line of lines) {
				if (line.startsWith('event: ')) {
					currentEvent = line.slice(7).trim() as SSEEventType;
				} else if (line.startsWith('data: ')) {
					currentData = line.slice(6);
				} else if (line === '' && currentEvent && currentData) {
					// Empty line marks end of event
					try {
						const parsedData = JSON.parse(currentData);
						yield { event: currentEvent, data: parsedData } as SSEEvent;
					} catch {
						console.warn('Failed to parse SSE data:', currentData);
					}
					currentEvent = null;
					currentData = '';
				}
			}
		}
	} finally {
		reader.releaseLock();
	}
}

/**
 * Send a non-streaming query to the RAG server.
 * Returns the complete response at once.
 */
export async function sendQuery(query: string, sessionId?: string): Promise<QueryResponse> {
	const response = await fetch(`${API_BASE}/query`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({ query, session_id: sessionId })
	});

	if (!response.ok) {
		throw new Error(`Query failed: ${response.statusText}`);
	}

	return response.json();
}

/**
 * Get chat history for a session.
 */
export async function getChatHistory(sessionId: string): Promise<ChatHistoryResponse> {
	const response = await fetch(`${API_BASE}/chat/history/${sessionId}`);
	if (!response.ok) {
		throw new Error(`Failed to fetch chat history: ${response.statusText}`);
	}
	return response.json();
}

/**
 * Clear chat history for a session.
 */
export async function clearChatSession(sessionId: string): Promise<void> {
	const response = await fetch(`${API_BASE}/chat/clear`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({ session_id: sessionId })
	});

	if (!response.ok) {
		throw new Error(`Failed to clear session: ${response.statusText}`);
	}
}

/**
 * Get the download URL for a document.
 */
export function getDocumentDownloadUrl(documentId: string): string {
	return `${API_BASE}/documents/${documentId}/download`;
}

// ============================================================================
// Types - Session Management
// ============================================================================

export interface SessionMetadata {
	session_id: string;
	title: string;
	created_at: string;
	updated_at: string;
	is_archived: boolean;
	is_temporary: boolean;
	llm_model?: string;
	search_type?: string; // "vector" | "hybrid"
}

export interface SessionListResponse {
	sessions: SessionMetadata[];
	total: number;
}

export interface CreateSessionRequest {
	is_temporary?: boolean;
	title?: string;
	first_message?: string;
}

export interface CreateSessionResponse {
	session_id: string;
	title: string;
	created_at: string;
	is_temporary: boolean;
	llm_model?: string;
	search_type?: string; // "vector" | "hybrid"
}

// ============================================================================
// API Functions - Session Management
// ============================================================================

/**
 * List all chat sessions (excluding temporary)
 */
export async function fetchChatSessions(includeArchived: boolean = false): Promise<SessionMetadata[]> {
	const params = new URLSearchParams();
	params.set('include_archived', includeArchived.toString());

	const response = await fetch(`${API_BASE}/chat/sessions?${params}`);
	if (!response.ok) {
		throw new Error(`Failed to fetch sessions: ${response.statusText}`);
	}
	const data: SessionListResponse = await response.json();
	return data.sessions;
}

/**
 * Create new chat session
 * @param firstMessage - If provided, generates an AI title from this message
 */
export async function createNewSession(firstMessage?: string): Promise<CreateSessionResponse> {
	const body: CreateSessionRequest = {};

	if (firstMessage) {
		body.first_message = firstMessage;
	}

	const response = await fetch(`${API_BASE}/chat/sessions/new`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify(body)
	});

	if (!response.ok) {
		throw new Error(`Failed to create session: ${response.statusText}`);
	}

	return response.json();
}

/**
 * Delete chat session (metadata + messages)
 */
export async function deleteSession(sessionId: string): Promise<void> {
	const response = await fetch(`${API_BASE}/chat/sessions/${sessionId}`, {
		method: 'DELETE'
	});

	if (!response.ok) {
		throw new Error(`Failed to delete session: ${response.statusText}`);
	}
}

/**
 * Archive chat session
 */
export async function archiveSession(sessionId: string): Promise<void> {
	const response = await fetch(`${API_BASE}/chat/sessions/${sessionId}/archive`, {
		method: 'POST'
	});

	if (!response.ok) {
		throw new Error(`Failed to archive session: ${response.statusText}`);
	}
}

/**
 * Restore chat session from archive
 */
export async function unarchiveSession(sessionId: string): Promise<void> {
	const response = await fetch(`${API_BASE}/chat/sessions/${sessionId}/unarchive`, {
		method: 'POST'
	});

	if (!response.ok) {
		throw new Error(`Failed to unarchive session: ${response.statusText}`);
	}
}

// ============================================================================
// Types - Settings
// ============================================================================

export interface Settings {
	contextual_retrieval_enabled: boolean;
}

export interface SettingsUpdate {
	contextual_retrieval_enabled?: boolean;
}

// ============================================================================
// API Functions - Settings
// ============================================================================

export async function fetchSettings(): Promise<Settings> {
	const response = await fetch(`${API_BASE}/settings`);
	if (!response.ok) {
		throw new Error(`Failed to fetch settings: ${response.statusText}`);
	}
	return response.json();
}

export async function updateSettings(update: SettingsUpdate): Promise<Settings> {
	const response = await fetch(`${API_BASE}/settings`, {
		method: 'PATCH',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(update)
	});
	if (!response.ok) {
		throw new Error(`Failed to update settings: ${response.statusText}`);
	}
	return response.json();
}

// ============================================================================
// Types - API Keys
// ============================================================================

export interface ApiKeyStatus {
	provider: string;
	has_key: boolean;
	masked_key: string | null;
}

export interface ApiKeyValidationError {
	detail: string;
}

export interface ApiKeySetResponse {
	provider: string;
	status: string;
	masked_key: string;
}

// ============================================================================
// API Functions - API Keys
// ============================================================================

/**
 * Fetch the status of all API keys (which providers need keys, which have them set).
 */
export async function fetchApiKeyStatus(): Promise<ApiKeyStatus[]> {
	const response = await fetch(`${API_BASE}/api-keys`);
	if (!response.ok) {
		throw new Error(`Failed to fetch API key status: ${response.statusText}`);
	}
	return response.json();
}

/**
 * Set and validate an API key for a provider.
 * Throws an error with validation message if the key is invalid.
 */
export async function setApiKey(provider: string, apiKey: string): Promise<ApiKeySetResponse> {
	const response = await fetch(`${API_BASE}/api-keys/${provider}`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({ api_key: apiKey })
	});

	if (!response.ok) {
		const error: ApiKeyValidationError = await response.json();
		throw new Error(error.detail || 'Failed to set API key');
	}

	return response.json();
}
