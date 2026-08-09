import { AlertCircle, Loader2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { apiFileUrl, getApiErrorMessage, getInvoiceResult } from "../api/client";
import AppShell from "../components/AppShell";
import InvoicePreview from "../components/InvoicePreview";
import JsonRaw from "../components/JsonRaw";
import PageIntro from "../components/PageIntro";
import ResultSummary from "../components/ResultSummary";
import type { ExtractResponse } from "../types/invoice";

type Tab = "preview" | "raw";

export default function ResultPage() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const initialResult = (location.state as { result?: ExtractResponse } | null)?.result;
  const [result, setResult] = useState<ExtractResponse | null>(initialResult || null);
  const [loading, setLoading] = useState(!initialResult);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("preview");
  const [selectedPage, setSelectedPage] = useState(0);

  useEffect(() => {
    if (result || !jobId) {
      return;
    }

    let cancelled = false;
    setLoading(true);
    getInvoiceResult(jobId)
      .then((payload) => {
        if (!cancelled) {
          setResult(payload);
        }
      })
      .catch((requestError) => {
        if (!cancelled) {
          setError(getApiErrorMessage(requestError));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [jobId, result]);

  const currentPage = useMemo(() => {
    if (!result?.pages.length) {
      return null;
    }
    return result.pages[Math.min(selectedPage, result.pages.length - 1)];
  }, [result, selectedPage]);

  const downloadJson = () => {
    if (!result) {
      return;
    }
    const anchor = document.createElement("a");
    anchor.href = apiFileUrl(result.download_url);
    anchor.download = `${result.filename.replace(/\.pdf$/i, "")}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  };

  return (
    <AppShell>
      <PageIntro />

      {loading && (
        <div className="flex items-center gap-3 rounded border border-line bg-white px-5 py-4 text-muted">
          <Loader2 size={20} className="animate-spin" />
          Đang tải kết quả OCR...
        </div>
      )}

      {error && (
        <div className="flex items-start gap-3 rounded border border-[#efb5b5] bg-[#fff4f4] px-5 py-4 text-[#9b2525]">
          <AlertCircle size={22} className="mt-0.5 shrink-0" />
          <p className="text-sm font-semibold leading-6">{error}</p>
        </div>
      )}

      {result && currentPage && (
        <>
          <ResultSummary
            result={result}
            onDownload={downloadJson}
            onReset={() => navigate("/")}
          />

          <div className="mb-6 flex flex-wrap items-end justify-between gap-4 border-b border-line">
            <div className="flex gap-1">
              <TabButton active={activeTab === "preview"} onClick={() => setActiveTab("preview")}>
                Xem trước dữ liệu
              </TabButton>
              <TabButton active={activeTab === "raw"} onClick={() => setActiveTab("raw")}>
                JSON raw
              </TabButton>
            </div>

            {result.pages.length > 1 && (
              <div className="mb-2 flex items-center gap-2">
                {result.pages.map((page, index) => (
                  <button
                    key={page.page}
                    type="button"
                    onClick={() => setSelectedPage(index)}
                    className={`h-9 rounded border px-3 font-mono text-sm font-semibold ${
                      selectedPage === index
                        ? "border-brand bg-brand text-white"
                        : "border-line bg-white text-muted"
                    }`}
                  >
                    {page.page}
                  </button>
                ))}
              </div>
            )}
          </div>

          {activeTab === "preview" ? (
            <InvoicePreview
              data={currentPage.data}
              pageNumber={currentPage.page}
              pageCount={result.page_count}
            />
          ) : (
            <JsonRaw result={result} />
          )}
        </>
      )}
    </AppShell>
  );
}

function TabButton({
  active,
  children,
  onClick,
}: {
  active: boolean;
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`border-b-2 px-5 py-4 font-mono text-sm font-bold transition ${
        active
          ? "border-brand text-brand"
          : "border-transparent text-muted hover:border-line hover:text-ink"
      }`}
    >
      {children}
    </button>
  );
}
