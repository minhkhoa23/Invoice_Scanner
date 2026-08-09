import {
  Archive,
  FileText,
  Loader2,
  Search,
  Upload,
  Zap,
} from "lucide-react";
import { useRef, useState } from "react";

interface UploadPanelProps {
  loading: boolean;
  progress: number;
  onFileSelected: (file: File) => void;
}

const MAX_SIZE_MB = 20;

function isPdf(file: File) {
  return file.name.toLowerCase().endsWith(".pdf") || file.type === "application/pdf";
}

export default function UploadPanel({
  loading,
  progress,
  onFileSelected,
}: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const acceptFile = (file?: File) => {
    if (!file) {
      return;
    }
    if (!isPdf(file)) {
      setLocalError("Chỉ hỗ trợ file PDF.");
      return;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setLocalError(`File vượt quá ${MAX_SIZE_MB} MB.`);
      return;
    }
    setLocalError(null);
    onFileSelected(file);
  };

  return (
    <section className="w-full max-w-[990px]">
      <div
        className={`relative grid min-h-[350px] place-items-center border-2 border-dashed bg-white transition ${
          dragging ? "border-brand shadow-soft" : "border-[#aaa7a1]"
        } ${loading ? "pointer-events-none" : ""}`}
        onDragEnter={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          event.preventDefault();
          setDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          acceptFile(event.dataTransfer.files[0]);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          className="sr-only"
          onChange={(event) => acceptFile(event.target.files?.[0])}
        />

        <div className="flex flex-col items-center px-6 text-center">
          <div className="relative mb-8 grid h-[86px] w-[72px] place-items-center rounded-md border border-[#aaa7a1] bg-white">
            <FileText size={42} className="text-[#918d86]" strokeWidth={1.5} />
            <span className="absolute -right-1 top-0 rounded-sm bg-[#dc2028] px-2 py-1 font-mono text-[10px] font-bold text-white">
              PDF
            </span>
          </div>

          <p className="text-[22px] font-semibold leading-8">
            {loading ? "Đang trích xuất dữ liệu..." : "Kéo thả file PDF hoặc "}
            {!loading && (
              <button
                type="button"
                className="font-semibold text-brand"
                onClick={() => inputRef.current?.click()}
              >
                chọn từ máy tính
              </button>
            )}
          </p>
          <p className="mt-2 text-[17px] text-muted">Chỉ hỗ trợ định dạng .pdf - tối đa 20 MB</p>

          {loading && (
            <div className="mt-8 w-full max-w-[360px]">
              <div className="h-2 overflow-hidden rounded-full bg-[#e9e7e3]">
                <div
                  className="h-full rounded-full bg-brand transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className="mt-3 flex items-center justify-center gap-2 font-mono text-sm text-muted">
                <Loader2 size={16} className="animate-spin" />
                {progress < 45 ? "Đang tải file" : "Đang OCR bằng Vintern"}
              </div>
            </div>
          )}

          {localError && <p className="mt-6 text-sm font-semibold text-[#c83232]">{localError}</p>}
        </div>
      </div>

      <div className="grid border-x border-b border-[#e4e1dc] bg-white sm:grid-cols-3">
        <FeatureItem icon={<Zap size={23} />} iconClass="text-[#ffc526]" label="Xử lý trong vài giây" />
        <FeatureItem
          icon={<Search size={25} />}
          iconClass="text-ink"
          label="Độ chính xác OCR > 95%"
          bordered
        />
        <FeatureItem
          icon={<Archive size={24} />}
          iconClass="text-[#b0784c]"
          label="Xuất JSON chuẩn hóa"
        />
      </div>
    </section>
  );
}

function FeatureItem({
  icon,
  iconClass,
  label,
  bordered = false,
}: {
  icon: React.ReactNode;
  iconClass: string;
  label: string;
  bordered?: boolean;
}) {
  return (
    <div
      className={`flex min-h-[88px] items-center gap-5 px-8 font-mono text-[17px] text-muted ${
        bordered ? "border-y border-[#e4e1dc] sm:border-x sm:border-y-0" : ""
      }`}
    >
      <span className={iconClass}>{icon}</span>
      <span className="leading-6">{label}</span>
    </div>
  );
}
