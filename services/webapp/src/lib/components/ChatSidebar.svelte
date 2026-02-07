<script lang="ts">
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import {
    sidebarOpen,
    toggleSidebar,
    sessionRefreshTrigger,
    showRecentExpanded,
    showArchivedExpanded,
    sidebarWidth,
    SIDEBAR_MIN_WIDTH,
    SIDEBAR_COLLAPSED_WIDTH
  } from '$lib/stores/sidebar';
  import {
    fetchChatSessions,
    deleteSession,
    archiveSession,
    unarchiveSession,
    getChatHistory
  } from '$lib/api';
  import type { SessionMetadata } from '$lib/api';
  import { onMount } from 'svelte';
  import ThemeToggle from './ThemeToggle.svelte';

  let activeSessions: SessionMetadata[] = $state([]);
  let archivedSessions: SessionMetadata[] = $state([]);
  let loading = $state(true);
  let error: string | null = $state(null);

  // Resize state
  let isResizing = $state(false);
  let maxWidth = $state(400);

  // Calculate max width as 1/3 of window
  function updateMaxWidth() {
    if (typeof window !== 'undefined') {
      maxWidth = Math.floor(window.innerWidth / 3);
    }
  }

  onMount(() => {
    loadSessions();
    updateMaxWidth();
    window.addEventListener('resize', updateMaxWidth);
    return () => window.removeEventListener('resize', updateMaxWidth);
  });

  // Handle resize drag
  function handleResizeStart(e: MouseEvent) {
    e.preventDefault();
    isResizing = true;
    document.addEventListener('mousemove', handleResizeMove);
    document.addEventListener('mouseup', handleResizeEnd);
  }

  function handleResizeMove(e: MouseEvent) {
    if (!isResizing) return;
    const newWidth = Math.min(Math.max(e.clientX, SIDEBAR_MIN_WIDTH), maxWidth);
    sidebarWidth.set(newWidth);
  }

  function handleResizeEnd() {
    isResizing = false;
    document.removeEventListener('mousemove', handleResizeMove);
    document.removeEventListener('mouseup', handleResizeEnd);
  }

  // Computed width style
  const sidebarStyle = $derived(
    $sidebarOpen
      ? `width: ${$sidebarWidth}px; min-width: ${$sidebarWidth}px;`
      : `width: ${SIDEBAR_COLLAPSED_WIDTH}px; min-width: ${SIDEBAR_COLLAPSED_WIDTH}px;`
  );

  // Watch for session refresh trigger (e.g., after first message creates a session)
  $effect(() => {
    const _ = $sessionRefreshTrigger;
    if (_ > 0) {
      loadSessions();
    }
  });

  async function loadSessions() {
    loading = true;
    error = null;
    try {
      const sessions = await fetchChatSessions(true);
      activeSessions = sessions.filter(s => !s.is_archived);
      archivedSessions = sessions.filter(s => s.is_archived);
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load sessions';
      console.error('Failed to load sessions:', err);
    } finally {
      loading = false;
    }
  }

  async function handleDeleteSession(sessionId: string) {
    try {
      await deleteSession(sessionId);
      await loadSessions();

      if ($page.url.searchParams.get('session_id') === sessionId) {
        await goto('/chat');
      }
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to delete session';
      console.error('Failed to delete session:', err);
    }
  }

  async function handleArchiveSession(sessionId: string) {
    try {
      await archiveSession(sessionId);
      await loadSessions();
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to archive session';
      console.error('Failed to archive session:', err);
    }
  }

  async function handleUnarchiveSession(sessionId: string) {
    try {
      await unarchiveSession(sessionId);
      await loadSessions();
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to unarchive session';
      console.error('Failed to unarchive session:', err);
    }
  }

  async function handleExportSession(sessionId: string, title: string) {
    try {
      const history = await getChatHistory(sessionId);
      if (history.messages.length === 0) {
        error = 'No messages to export';
        return;
      }

      const text = history.messages
        .map(m => `${m.role === 'user' ? 'User' : 'Assistant'}:\n${m.content}`)
        .join('\n\n---\n\n');

      const blob = new Blob([text], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${title || 'chat'}-${sessionId.slice(0, 8)}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to export session';
      console.error('Failed to export session:', err);
    }
  }

  function handleSessionClick(sessionId: string) {
    goto(`/chat?session_id=${sessionId}`);
  }

  function isCurrentSession(sessionId: string): boolean {
    return $page.url.searchParams.get('session_id') === sessionId;
  }
</script>

<aside
  class="sidebar-container bg-base-200 border-r border-base-300 flex flex-col h-screen relative"
  class:resizing={isResizing}
  style={sidebarStyle}
>
  <!-- Header with logo and toggle -->
  <div class="p-3 flex items-center" class:justify-between={$sidebarOpen} class:justify-center={!$sidebarOpen}>
    {#if $sidebarOpen}
      <a href="/" class="flex items-center gap-2 hover:opacity-80 transition-opacity">
        <img src="/binchicken.png" alt="RAG Bench" class="h-7 w-7" />
        <span class="text-lg font-semibold text-base-content">RAG Lab</span>
      </a>
    {/if}
    <button
      class="btn btn-ghost btn-square btn-sm"
      onclick={toggleSidebar}
      aria-label={$sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
      title={$sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
    >
      <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
        <rect x="3" y="3" width="7" height="18" rx="1" />
        <rect x="14" y="3" width="7" height="18" rx="1" />
      </svg>
    </button>
  </div>

  <!-- Menu Items - always visible -->
  <div class="px-2 pb-2" class:px-3={$sidebarOpen}>
    <!-- Chat -->
    <a
      href="/chat"
      class="menu-item flex items-center w-full p-2 rounded-lg hover:bg-base-300 transition-colors text-base-content"
      class:gap-3={$sidebarOpen}
      class:justify-center={!$sidebarOpen}
      title={!$sidebarOpen ? 'Chat' : undefined}
    >
      <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
        <path stroke-linecap="round" stroke-linejoin="round" d="M8.625 12a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H8.25m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H12m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 0 1-2.555-.337A5.972 5.972 0 0 1 5.41 20.97a5.969 5.969 0 0 1-.474-.065 4.48 4.48 0 0 0 .978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25Z" />
      </svg>
      {#if $sidebarOpen}
        <span class="text-sm truncate">Chat</span>
      {/if}
    </a>

    <!-- Documents -->
    <a
      href="/documents"
      class="menu-item flex items-center w-full p-2 rounded-lg hover:bg-base-300 transition-colors text-base-content"
      class:gap-3={$sidebarOpen}
      class:justify-center={!$sidebarOpen}
      title={!$sidebarOpen ? 'Documents' : undefined}
    >
      <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
        <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
      </svg>
      {#if $sidebarOpen}
        <span class="text-sm truncate">Documents</span>
      {/if}
    </a>

    <!-- Analytics -->
    <a
      href="/analytics"
      class="menu-item flex items-center w-full p-2 rounded-lg hover:bg-base-300 transition-colors text-base-content"
      class:gap-3={$sidebarOpen}
      class:justify-center={!$sidebarOpen}
      title={!$sidebarOpen ? 'Analytics' : undefined}
    >
      <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
        <path stroke-linecap="round" stroke-linejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
      </svg>
      {#if $sidebarOpen}
        <span class="text-sm truncate">Analytics</span>
      {/if}
    </a>

    <!-- Settings -->
    <a
      href="/settings"
      class="menu-item flex items-center w-full p-2 rounded-lg hover:bg-base-300 transition-colors text-base-content"
      class:gap-3={$sidebarOpen}
      class:justify-center={!$sidebarOpen}
      title={!$sidebarOpen ? 'Settings' : undefined}
    >
      <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
        <path stroke-linecap="round" stroke-linejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 0 1 1.37.49l1.296 2.247a1.125 1.125 0 0 1-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 0 1 0 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 0 1-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 0 1-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 0 1-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 0 1-1.369-.49l-1.297-2.247a1.125 1.125 0 0 1 .26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 0 1 0-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 0 1-.26-1.43l1.297-2.247a1.125 1.125 0 0 1 1.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28Z" />
        <path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
      </svg>
      {#if $sidebarOpen}
        <span class="text-sm truncate">Settings</span>
      {/if}
    </a>

    <!-- Divider between menu and sessions -->
    {#if $sidebarOpen}
      <div class="h-4"></div>
    {/if}
  </div>

  <!-- Expanded content -->
  {#if $sidebarOpen}
    <div class="flex-1 flex flex-col overflow-hidden sidebar-content">

      <!-- Error Display -->
      {#if error}
        <div class="alert alert-error mx-3 mb-3 text-sm py-2">
          <span>{error}</span>
        </div>
      {/if}

      <!-- Sessions List - resizes with panel -->
      <div class="flex-1 overflow-y-auto px-3">
        {#if loading}
          <div class="flex items-center justify-center py-8">
            <span class="loading loading-spinner loading-md"></span>
          </div>
        {:else}
          <!-- Recent (Active) Sessions -->
          <div class="mb-4">
            <button
              class="w-full flex items-center justify-between text-xs font-semibold text-base-content/60 uppercase tracking-wider hover:text-base-content mb-2"
              onclick={() => showRecentExpanded.update(v => !v)}
            >
              <span>Recent</span>
              <div class="flex items-center gap-1">
                <span class="text-xs text-base-content/40 normal-case">{activeSessions.length}</span>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  class="h-3 w-3 transition-transform {$showRecentExpanded ? 'rotate-180' : ''}"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                </svg>
              </div>
            </button>

            {#if $showRecentExpanded}
              {#if activeSessions.length === 0}
                <p class="text-xs text-base-content/50 text-center py-3">No active chats</p>
              {:else}
                <div class="space-y-0.5">
                  {#each activeSessions as session (session.session_id)}
                    <div
                      class="session-item group relative py-1.5 px-2 rounded-lg hover:bg-base-300 cursor-pointer transition-colors {isCurrentSession(session.session_id) ? 'bg-primary/10 border border-primary/30' : ''}"
                      onclick={() => handleSessionClick(session.session_id)}
                      onkeydown={(e) => e.key === 'Enter' && handleSessionClick(session.session_id)}
                      role="button"
                      tabindex="0"
                    >
                      <p class="session-title text-sm truncate pr-1">
                        {session.title || 'Untitled Chat'}
                      </p>

                      <!-- Action buttons - overlay on hover -->
                      <div class="session-actions absolute right-1 top-1/2 -translate-y-1/2 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity bg-base-300 rounded-lg px-1 py-0.5">
                        <div class="tooltip tooltip-right" data-tip="Archive">
                          <button
                            class="btn btn-ghost btn-xs btn-square"
                            onclick={(e) => { e.stopPropagation(); handleArchiveSession(session.session_id); }}
                            aria-label="Archive session"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                            </svg>
                          </button>
                        </div>
                        <div class="tooltip tooltip-right" data-tip="Delete">
                          <button
                            class="btn btn-ghost btn-xs btn-square text-error hover:bg-error/20"
                            onclick={(e) => { e.stopPropagation(); handleDeleteSession(session.session_id); }}
                            aria-label="Delete session"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                          </button>
                        </div>
                        <div class="tooltip tooltip-right" data-tip="Export">
                          <button
                            class="btn btn-ghost btn-xs btn-square"
                            onclick={(e) => { e.stopPropagation(); handleExportSession(session.session_id, session.title); }}
                            aria-label="Export chat"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                            </svg>
                          </button>
                        </div>
                      </div>
                    </div>
                  {/each}
                </div>
              {/if}
            {/if}
          </div>

          <!-- Archived Sessions -->
          <div class="border-t border-base-300 pt-3">
            <button
              class="w-full flex items-center justify-between text-xs font-semibold text-base-content/60 uppercase tracking-wider hover:text-base-content mb-2"
              onclick={() => showArchivedExpanded.update(v => !v)}
            >
              <span>Archived</span>
              <div class="flex items-center gap-1">
                <span class="text-xs text-base-content/40 normal-case">{archivedSessions.length}</span>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  class="h-3 w-3 transition-transform {$showArchivedExpanded ? 'rotate-180' : ''}"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                </svg>
              </div>
            </button>

            {#if $showArchivedExpanded}
              {#if archivedSessions.length === 0}
                <p class="text-xs text-base-content/50 text-center py-3">No archived chats</p>
              {:else}
                <div class="space-y-0.5">
                  {#each archivedSessions as session (session.session_id)}
                    <div
                      class="session-item group relative py-1.5 px-2 rounded-lg hover:bg-base-300 cursor-pointer transition-colors"
                      onclick={() => handleSessionClick(session.session_id)}
                      onkeydown={(e) => e.key === 'Enter' && handleSessionClick(session.session_id)}
                      role="button"
                      tabindex="0"
                    >
                      <p class="session-title text-sm truncate opacity-70 pr-1">
                        {session.title || 'Untitled Chat'}
                      </p>

                      <!-- Action buttons - overlay on hover -->
                      <div class="session-actions absolute right-1 top-1/2 -translate-y-1/2 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity bg-base-300 rounded-lg px-1 py-0.5">
                        <div class="tooltip tooltip-right" data-tip="Unarchive">
                          <button
                            class="btn btn-ghost btn-xs btn-square"
                            onclick={(e) => { e.stopPropagation(); handleUnarchiveSession(session.session_id); }}
                            aria-label="Unarchive session"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
                            </svg>
                          </button>
                        </div>
                        <div class="tooltip tooltip-right" data-tip="Delete">
                          <button
                            class="btn btn-ghost btn-xs btn-square text-error hover:bg-error/20"
                            onclick={(e) => { e.stopPropagation(); handleDeleteSession(session.session_id); }}
                            aria-label="Delete session"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                          </button>
                        </div>
                        <div class="tooltip tooltip-right" data-tip="Export">
                          <button
                            class="btn btn-ghost btn-xs btn-square"
                            onclick={(e) => { e.stopPropagation(); handleExportSession(session.session_id, session.title); }}
                            aria-label="Export chat"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                            </svg>
                          </button>
                        </div>
                      </div>
                    </div>
                  {/each}
                </div>
              {/if}
            {/if}
          </div>
        {/if}
      </div>

    </div>
  {/if}

  <!-- Theme toggle at bottom - always visible -->
  <div class="mt-auto p-2 border-t border-base-300" class:p-3={$sidebarOpen}>
    <div class="flex items-center" class:gap-2={$sidebarOpen} class:justify-center={!$sidebarOpen}>
      <ThemeToggle />
      {#if $sidebarOpen}
        <span class="text-sm text-base-content/70">Theme</span>
      {/if}
    </div>
  </div>

  <!-- Resize handle - only when expanded -->
  {#if $sidebarOpen}
    <div
      class="resize-handle"
      onmousedown={handleResizeStart}
      role="separator"
      aria-label="Resize sidebar"
      tabindex="0"
    ></div>
  {/if}
</aside>

<style>
  .sidebar-container {
    transition: width 0.2s ease, min-width 0.2s ease;
    overflow: hidden;
  }

  .sidebar-container.resizing {
    transition: none;
    user-select: none;
  }

  .sidebar-content {
    opacity: 0;
    animation: fadeIn 0.2s ease 0.1s forwards;
    overflow: hidden;
  }

  /* Menu items truncate text with ellipsis at panel edge */
  .menu-item {
    overflow: hidden;
    min-width: 0;
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }

  .resize-handle {
    position: absolute;
    top: 0;
    right: 0;
    width: 4px;
    height: 100%;
    cursor: ew-resize;
    background: transparent;
    transition: background 0.2s ease;
  }

  .resize-handle:hover,
  .sidebar-container.resizing .resize-handle {
    background: oklch(var(--p) / 0.3);
  }
</style>
