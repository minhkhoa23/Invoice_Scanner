import type { ExtractResponse } from "../types/invoice";

export default function JsonRaw({ result }: { result: ExtractResponse }) {
  return (
    <pre className="max-h-[780px] overflow-auto rounded border border-line bg-[#171717] p-6 font-mono text-sm leading-6 text-[#e9e7e3]">
      {JSON.stringify(result.pages, null, 2)}
    </pre>
  );
}
