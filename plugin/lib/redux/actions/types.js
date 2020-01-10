'use babel';

function generateActionType(namespace, action) {
  return namespace + ':' + action;
}

function generateActionNamespaceChecker(namespace) {
  return function(candidateAction) {
    return candidateAction.type.split(':')[0] === namespace;
  };
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
export const CONN_INITIATED = generateActionType(CONN_NAMESPACE, 'initiated');
// The connection to the server has been established and initialized
export const CONN_INITIALIZED = generateActionType(CONN_NAMESPACE, 'initialized');
// We failed to establish and initialize a connection with the server in time
export const CONN_TIMEOUT = generateActionType(CONN_NAMESPACE, 'timeout');
// We received an error while connecting to or intializing a connection with the server
export const CONN_ERROR = generateActionType(CONN_NAMESPACE, 'error');
// We lost our connection to the server (possibly because the server was shut down)
export const CONN_LOST = generateActionType(CONN_NAMESPACE, 'lost');

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
