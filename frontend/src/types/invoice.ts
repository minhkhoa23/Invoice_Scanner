export type Nullable<T> = T | null;

export interface InvoiceInfo {
  invoice_type: Nullable<string>;
  invoice_number: Nullable<string>;
  series: Nullable<string>;
  invoice_date: Nullable<string>;
  tax_authority_code: Nullable<string>;
  currency: Nullable<string>;
  payment_method: Nullable<string>;
}

export interface PartyInfo {
  name: Nullable<string>;
  english_name?: Nullable<string>;
  company_name?: Nullable<string>;
  tax_code: Nullable<string>;
  address: Nullable<string>;
  phone?: string[];
  fax?: string[];
  email?: Nullable<string>;
  website?: Nullable<string>;
  id_card?: Nullable<string>;
  passport_number?: Nullable<string>;
  account_number?: Nullable<string>;
  budgetary_unit_code?: Nullable<string>;
}

export interface InvoiceItem {
  line_number: Nullable<number>;
  description: Nullable<string>;
  description_lines?: string[];
  item_type?: Nullable<string>;
  container_number?: Nullable<string>;
  unit: Nullable<string>;
  quantity: Nullable<number>;
  unit_price: Nullable<number>;
  amount: Nullable<number>;
  taxable_amount?: Nullable<number>;
  vat_rate?: Nullable<number>;
  vat_amount?: Nullable<number>;
  total_amount?: Nullable<number>;
}

export interface TotalsInfo {
  subtotal: Nullable<number>;
  vat_rate: Nullable<number>;
  vat_amount: Nullable<number>;
  total_payment: Nullable<number>;
  amount_in_words: Nullable<string>;
}

export interface InvoiceData {
  invoice: InvoiceInfo;
  seller: PartyInfo;
  buyer: PartyInfo;
  shipping: {
    ship_from_warehouse: Nullable<string>;
  };
  items: InvoiceItem[];
  totals: TotalsInfo;
  signature: {
    is_valid: Nullable<boolean>;
    signed_by: Nullable<string>;
    signed_date: Nullable<string>;
  };
  metadata: {
    source_type: Nullable<string>;
    page_count: Nullable<number>;
    ocr_processed: boolean;
    pdf_text_layer_used?: Nullable<boolean>;
    page_image_path?: string;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface PageResult {
  page: number;
  data: InvoiceData;
  raw_response?: string;
}

export interface ExtractResponse {
  job_id: string;
  filename: string;
  page_count: number;
  source_type: Nullable<string>;
  confidence: number;
  elapsed_seconds: number;
  pages: PageResult[];
  data: InvoiceData;
  download_url: string;
}

export interface ServerStatus {
  server_url: string;
  origin: string;
  reachable: boolean;
  checks: Array<Record<string, unknown>>;
  server_command: string;
}
