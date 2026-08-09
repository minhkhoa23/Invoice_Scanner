export default function PageIntro() {
  return (
    <section className="mb-12">
      <h1 className="text-[34px] font-bold leading-tight tracking-normal text-ink sm:text-[40px]">
        Trích xuất dữ liệu hóa đơn
      </h1>
      <p className="mt-4 max-w-[620px] text-[18px] leading-8 text-muted">
        Tải lên file PDF hóa đơn - hệ thống sẽ nhận diện OCR và trả về dữ liệu có
        cấu trúc dạng JSON.
      </p>
    </section>
  );
}
