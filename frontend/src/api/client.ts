import axios, { AxiosError, type AxiosProgressEvent } from "axios";
import type { ExtractResponse, ServerStatus } from "../types/invoice";

export const API_BASE_URL =
  import.meta.env.VITE_API_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 700_000,
});

export function apiFileUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) {
    return path;
  }
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

export async function extractInvoice(
  file: File,
  onUploadProgress?: (event: AxiosProgressEvent) => void,
): Promise<ExtractResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post<ExtractResponse>("/api/invoices/extract", formData, {
    onUploadProgress,
  });
  return response.data;
}

export async function getInvoiceResult(jobId: string): Promise<ExtractResponse> {
  const response = await api.get<ExtractResponse>(`/api/invoices/${jobId}`);
  return response.data;
}

export async function getServerStatus(): Promise<ServerStatus> {
  const response = await api.get<ServerStatus>("/api/status");
  return response.data;
}

export function getApiErrorMessage(error: unknown): string {
  const axiosError = error as AxiosError<{ detail?: string }>;
  if (axiosError.response?.data?.detail) {
    return axiosError.response.data.detail;
  }
  if (axiosError.message) {
    return axiosError.message;
  }
  return "Không thể xử lý hóa đơn. Vui lòng thử lại.";
}
