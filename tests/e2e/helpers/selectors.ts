/**
 * Seletores centralizados para componentes Streamlit.
 * Streamlit nesta versão renderiza diretamente na página (sem iframe).
 */

// File upload
export const FILE_UPLOADER = '[data-testid="stFileUploader"]';
export const FILE_UPLOADER_INPUT = '[data-testid="stFileUploaderDropzone"] input[type="file"]';
export const FILE_UPLOADER_DELETE = '[data-testid="stFileUploaderDeleteBtn"]';

// Buttons
export const PRIMARY_BUTTON = 'button[kind="primary"]';
export const DOWNLOAD_BUTTON = '[data-testid="stDownloadButton"]';

// Layout
export const SIDEBAR = '[data-testid="stSidebar"]';
export const MAIN_CONTENT = '[data-testid="stMain"]';

// Content
export const MARKDOWN = '[data-testid="stMarkdown"]';
export const EXPANDER = '[data-testid="stExpander"]';
export const ALERT_ERROR = '[data-testid="stAlert"][data-type="error"]';
export const ALERT_WARNING = '[data-testid="stAlert"][data-type="warning"]';
export const ALERT_INFO = '[data-testid="stAlert"][data-type="info"]';
export const METRIC = '[data-testid="stMetric"]';
export const IMAGE = '[data-testid="stImage"]';
export const CHAT_MESSAGE = '[data-testid="stChatMessage"]';

// Status/spinner
export const SPINNER = '[data-testid="stSpinner"]';
export const STATUS = '[data-testid="stStatus"]';
