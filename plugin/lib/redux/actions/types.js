'use babel';

function generateActionType(namespace, action) {
  return namespace + ':' + action;
}

function generateActionNamespaceChecker(namespace) {
  return function(candidateNamespace) {
    return candidateNamespace === namespace;
  };
}

export function getActionNamespace(action) {
  return action.type.split(':')[0];
}

// ============================================================================

// App-related Actions
const APP_NAMESPACE = 'app';
export const isAppAction = generateActionNamespaceChecker(APP_NAMESPACE);

// The user "opened" Skyline (i.e. the Skyline sidebar becomes visible)
export const APP_OPENED = generateActionType(APP_NAMESPACE, 'opened');
// The user "closed" Skyline (i.e. the user closed the Skyline sidebar)
export const APP_CLOSED = generateActionType(APP_NAMESPACE, 'closed');

// ============================================================================

// Connection-related Actions
const CONN_NAMESPACE = 'conn';
export const isConnectionAction = generateActionNamespaceChecker(CONN_NAMESPACE);

// We initiated a connection and we have not heard back from the server yet
export const CONN_CONNECTING = generateActionType(CONN_NAMESPACE, 'connecting');
// The connection to the server has been established and we are now waiting on initialization
export const CONN_INITIALIZING = generateActionType(CONN_NAMESPACE, 'initializing');
// The connection to the server has been established and initialized
export const CONN_INITIALIZED = generateActionType(CONN_NAMESPACE, 'initialized');
// We received an error while connecting to or intializing a connection with the server
export const CONN_ERROR = generateActionType(CONN_NAMESPACE, 'error');
// We lost our connection to the server (possibly because the server was shut down)
export const CONN_LOST = generateActionType(CONN_NAMESPACE, 'lost');
// Increment the sequence number tracker in the connection state
export const CONN_INCR_SEQ = generateActionType(CONN_NAMESPACE, 'incr_seq');

// ============================================================================

// Analysis-related Actions
const ANALYSIS_NAMESPACE = 'analysis';
export const isAnalysisAction = generateActionNamespaceChecker(ANALYSIS_NAMESPACE);

// We issued an analysis request
export const ANALYSIS_REQ = generateActionType(ANALYSIS_NAMESPACE, 'req');
// We received a memory breakdown
export const ANALYSIS_REC_MEM_BREAKDOWN = generateActionType(ANALYSIS_NAMESPACE, 'rec_mem_breakdown');
// We received memory usage information
export const ANALYSIS_REC_MEM_USAGE = generateActionType(ANALYSIS_NAMESPACE, 'rec_mem_usage');
// We received throughput information
export const ANALYSIS_REC_THPT = generateActionType(ANALYSIS_NAMESPACE, 'rec_thpt');
// An error occurred during the analysis
export const ANALYSIS_ERROR = generateActionType(ANALYSIS_NAMESPACE, 'error');

// ============================================================================

// Project-related Actions
const PROJECT_NAMESPACE = 'proj';
export const isProjectAction = generateActionNamespaceChecker(PROJECT_NAMESPACE);

// The project's modified status has changed (e.g., unmodified -> modified)
export const PROJECT_MODIFIED_CHANGE = generateActionType(PROJECT_NAMESPACE, 'modified_change');
// The Atom TextEditors associated with the relevant project files have changed
export const PROJECT_EDITORS_CHANGE = generateActionType(PROJECT_NAMESPACE, 'editors_change');
