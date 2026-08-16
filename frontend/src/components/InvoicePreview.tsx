import type { InvoiceData, InvoiceItem } from "../types/invoice";

interface InvoicePreviewProps {
  data: InvoiceData;
  pageNumber: number;
  pageCount: number;
  pageLabel?: string;
}

const empty = "-";

function display(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return empty;
  }
  if (Array.isArray(value)) {
    return value.length ? value.join(", ") : empty;
  }
  if (typeof value === "boolean") {
    return value ? "Có" : "Không";
  }
  return String(value);
}

function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return empty;
  }
  return new Intl.NumberFormat("vi-VN").format(value);
}

function formatMoney(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return empty;
  }
  return `${formatNumber(value)} đ`;
}

function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return empty;
  }
  return `${formatNumber(value)}%`;
}

export default function InvoicePreview({ data, pageNumber, pageCount, pageLabel }: InvoicePreviewProps) {
  const invoiceNumber = data.invoice?.invoice_number || "Chưa xác định";
  const total = data.totals?.total_payment;
  const items = data.items || [];

  return (
    <article className="rounded border border-line bg-white px-8 py-8 sm:px-9 sm:py-9">
      <div className="flex flex-col gap-5 border-b border-line pb-8 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="font-mono text-xs font-bold uppercase text-muted">Số hóa đơn</p>
          <h2 className="mt-3 font-mono text-[28px] font-bold leading-tight tracking-normal">
            {invoiceNumber}
          </h2>
          {pageCount > 1 && (
            <p className="mt-2 font-mono text-sm text-muted">
              {pageLabel || `Trang ${pageNumber}/${pageCount}`}
            </p>
          )}
        </div>
        <div className="text-left sm:text-right">
          <p className="font-mono text-xs font-bold uppercase text-muted">Tổng cộng</p>
          <p className="mt-3 text-[28px] font-bold leading-tight">{formatMoney(total)}</p>
        </div>
      </div>

      <div className="grid gap-8 border-b border-line py-8 lg:grid-cols-2">
        <Section title="Nhà cung cấp">
          <Field label="Tên công ty" value={data.seller?.name} />
          <Field label="Mã số thuế" value={data.seller?.tax_code} strong />
          <Field label="Địa chỉ" value={data.seller?.address} />
          <Field label="Email" value={data.seller?.email} strong />
          <Field label="Điện thoại" value={data.seller?.phone} strong />
        </Section>

        <Section title="Thông tin thanh toán">
          <Field label="Ngày xuất" value={data.invoice?.invoice_date} strong />
          <Field label="Phương thức" value={data.invoice?.payment_method} />
          <Field label="Đồng tiền" value={data.invoice?.currency || "VND"} strong />
          <Field label="Loại hóa đơn" value={data.invoice?.invoice_type} />
          <Field label="Ký hiệu" value={data.invoice?.series} strong />
        </Section>

        <Section title="Khách hàng">
          <Field label="Tên công ty" value={data.buyer?.company_name || data.buyer?.name} />
          <Field label="Mã số thuế" value={data.buyer?.tax_code} strong />
          <Field label="Địa chỉ" value={data.buyer?.address} />
          <Field label="Số tài khoản" value={data.buyer?.account_number} strong />
        </Section>

        <Section title="Xác thực">
          <Field label="Chữ ký hợp lệ" value={data.signature?.is_valid} strong />
          <Field label="Ký bởi" value={data.signature?.signed_by} />
          <Field label="Ngày ký" value={data.signature?.signed_date} />
          <Field label="Kho xuất" value={data.shipping?.ship_from_warehouse} />
        </Section>
      </div>

      <section className="py-8">
        <h3 className="border-b-4 border-ink pb-3 font-mono text-sm font-bold uppercase text-muted">
          Chi tiết dịch vụ / hàng hóa
        </h3>
        <div className="overflow-x-auto">
          <table className="mt-4 min-w-[1120px] w-full border-collapse text-left">
            <thead>
              <tr className="bg-[#f3f2f0] font-mono text-xs font-bold uppercase text-muted">
                <th className="px-3 py-3">Mô tả</th>
                <th className="px-3 py-3">Container</th>
                <th className="px-3 py-3 text-right">SL</th>
                <th className="px-3 py-3 text-right">Đơn vị</th>
                <th className="px-3 py-3 text-right">Đơn giá</th>
                <th className="px-3 py-3 text-right">Trước thuế</th>
                <th className="px-3 py-3 text-right">VAT</th>
                <th className="px-3 py-3 text-right">Tiền thuế</th>
                <th className="px-3 py-3 text-right">Tổng dòng</th>
              </tr>
            </thead>
            <tbody>
              {items.length ? (
                items.map((item, index) => <ItemRow item={item} key={`${item.line_number}-${index}`} />)
              ) : (
                <tr>
                  <td className="border-b border-line px-3 py-5 text-muted" colSpan={9}>
                    Không có dòng hàng hóa.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="ml-auto mt-7 w-full max-w-[360px] font-mono text-[15px]">
          <TotalRow label="Cộng tiền hàng" value={formatMoney(data.totals?.subtotal)} />
          <TotalRow
            label={`Thuế GTGT (${formatNumber(data.totals?.vat_rate)}%)`}
            value={formatMoney(data.totals?.vat_amount)}
          />
          <div className="mt-3 flex items-center justify-between rounded bg-ink px-4 py-4 font-bold text-white">
            <span>Tổng thanh toán</span>
            <span>{formatMoney(data.totals?.total_payment)}</span>
          </div>
        </div>

        {data.totals?.amount_in_words && (
          <p className="mt-6 text-right text-sm font-semibold text-muted">
            {data.totals.amount_in_words}
          </p>
        )}
      </section>
    </article>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h3 className="border-b-4 border-ink pb-3 font-mono text-sm font-bold uppercase text-muted">
        {title}
      </h3>
      <div>{children}</div>
    </section>
  );
}

function Field({
  label,
  value,
  strong = false,
}: {
  label: string;
  value: unknown;
  strong?: boolean;
}) {
  return (
    <div className="grid grid-cols-[150px_1fr] gap-5 border-b border-line py-4 text-[15px]">
      <span className="font-mono text-muted">{label}</span>
      <span className={strong ? "font-mono font-bold" : "font-medium"}>{display(value)}</span>
    </div>
  );
}

function ItemRow({ item }: { item: InvoiceItem }) {
  const taxableAmount = item.taxable_amount ?? item.amount;
  const lineTotal = item.total_amount ?? item.amount;

  return (
    <tr className="text-[15px]">
      <td className="border-b border-line px-3 py-4 font-medium">{display(item.description)}</td>
      <td className="border-b border-line px-3 py-4 font-mono text-muted">
        {display(item.container_number)}
      </td>
      <td className="border-b border-line px-3 py-4 text-right font-mono">{formatNumber(item.quantity)}</td>
      <td className="border-b border-line px-3 py-4 text-right text-muted">{display(item.unit)}</td>
      <td className="border-b border-line px-3 py-4 text-right font-mono">{formatNumber(item.unit_price)}</td>
      <td className="border-b border-line px-3 py-4 text-right font-mono">{formatNumber(taxableAmount)}</td>
      <td className="border-b border-line px-3 py-4 text-right font-mono">{formatPercent(item.vat_rate)}</td>
      <td className="border-b border-line px-3 py-4 text-right font-mono">{formatNumber(item.vat_amount)}</td>
      <td className="border-b border-line px-3 py-4 text-right font-mono font-bold">{formatNumber(lineTotal)}</td>
    </tr>
  );
}

function TotalRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between border-b border-line px-2 py-3 text-muted">
      <span>{label}</span>
      <span className="font-bold text-ink">{value}</span>
    </div>
  );
}
