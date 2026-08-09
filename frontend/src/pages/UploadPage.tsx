import type { AxiosProgressEvent } from "axios";
import { AlertCircle } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { extractInvoice, getApiErrorMessage } from "../api/client";
import AppShell from "../components/AppShell";
import PageIntro from "../components/PageIntro";
import UploadPanel from "../components/UploadPanel";

export default function UploadPage() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!loading) {
      return;
    }
    const timer = window.setInterval(() => {
      setProgress((current) => {
        if (current < 45) {
          return current;
        }
        return Math.min(92, current + 2);
      });
    }, 1200);
    return () => window.clearInterval(timer);
  }, [loading]);

  const handleFileSelected = async (file: File) => {
    setLoading(true);
    setError(null);
    setProgress(8);

    try {
      const result = await extractInvoice(file, (event: AxiosProgressEvent) => {
        if (!event.total) {
          return;
        }
        const percent = Math.round((event.loaded / event.total) * 40);
        setProgress(Math.max(8, Math.min(45, percent)));
        if (event.loaded >= event.total) {
          setProgress(52);
        }
      });
      setProgress(100);
      navigate(`/results/${result.job_id}`, { state: { result } });
    } catch (requestError) {
      setError(getApiErrorMessage(requestError));
      setProgress(0);
    } finally {
      setLoading(false);
    }
  };

  return (
    <AppShell>
      <PageIntro />

      {error && (
        <div className="mb-6 flex max-w-[990px] items-start gap-3 rounded border border-[#efb5b5] bg-[#fff4f4] px-5 py-4 text-[#9b2525]">
          <AlertCircle size={22} className="mt-0.5 shrink-0" />
          <p className="text-sm font-semibold leading-6">{error}</p>
        </div>
      )}

      <UploadPanel loading={loading} progress={progress} onFileSelected={handleFileSelected} />
    </AppShell>
  );
}
