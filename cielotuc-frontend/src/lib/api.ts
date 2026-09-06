export const API_BASE =
  (import.meta as unknown as { env: Record<string, string> }).env?.VITE_API_BASE_URL ??
  "http://127.0.0.1:8000";
