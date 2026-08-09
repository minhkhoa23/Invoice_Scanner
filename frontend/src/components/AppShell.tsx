import { FileText } from "lucide-react";
import { Link } from "react-router-dom";

interface AppShellProps {
  children: React.ReactNode;
}

export default function AppShell({ children }: AppShellProps) {
  return (
    <div className="min-h-screen bg-page text-ink">
      <header className="border-b border-line bg-white">
        <div className="mx-auto flex h-[74px] max-w-[1220px] items-center justify-between px-5 sm:px-8">
          <Link to="/" className="flex items-center gap-4" aria-label="invoiceOCR">
            <span className="grid h-10 w-10 place-items-center rounded-md bg-ink text-white shadow-soft">
              <FileText size={20} strokeWidth={2.2} />
            </span>
            <span className="text-[20px] font-bold tracking-normal">
              invoice<span className="text-brand">OCR</span>
            </span>
          </Link>
          <div className="font-mono text-sm font-semibold uppercase tracking-normal text-muted">
            PDF <span className="px-1">→</span> JSON
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-[1100px] px-5 pb-20 pt-14 sm:px-8 sm:pt-20">
        {children}
      </main>
    </div>
  );
}
