import { Check, Download, RefreshCw } from "lucide-react";
import type { ExtractResponse } from "../types/invoice";

interface ResultSummaryProps {
  result: ExtractResponse;
  onDownload: () => void;
  onReset: () => void;
}

export default function ResultSummary({ result, onDownload, onReset }: ResultSummaryProps) {
  return (
    <section className="mb-7 flex flex-col gap-5 rounded border border-line bg-white px-6 py-6 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-center gap-5">
        <span className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-success text-white">
          <Check size={28} strokeWidth={2.5} />
        </span>
        <div>
          <p className="text-[18px] font-bold">Trích xuất thành công</p>
          <p className="mt-1 font-mono text-[15px] text-muted">{result.filename}</p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <span className="rounded bg-[#dbf8e8] px-3 py-2 font-mono text-sm font-bold text-success">
          OCR {result.confidence}%
        </span>
        <button
          type="button"
          onClick={onDownload}
          className="inline-flex h-12 items-center gap-2 rounded bg-brand px-5 font-semibold text-white transition hover:bg-[#0c63df]"
        >
          <Download size={19} />
          Tải JSON
        </button>
        <button
          type="button"
          onClick={onReset}
          className="inline-flex h-12 items-center gap-2 rounded border border-[#b9b5ad] bg-white px-5 font-semibold text-muted transition hover:border-ink hover:text-ink"
        >
          <RefreshCw size={18} />
          Hóa đơn khác
        </button>
      </div>
    </section>
  );
}
