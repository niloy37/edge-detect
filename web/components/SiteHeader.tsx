import Link from "next/link";

const NAV = [
  { href: "/", label: "Demo" },
  { href: "/benchmarks", label: "Benchmarks" },
  { href: "/how-it-works", label: "How it works" },
];

export function SiteHeader({ active }: { active: string }) {
  return (
    <header className="border-b border-neutral-900">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
        <Link href="/" className="font-mono text-sm font-semibold tracking-tight text-neutral-100">
          edge<span className="text-neutral-600">-</span>detect
        </Link>
        <nav className="flex gap-4">
          {NAV.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={
                item.href === active
                  ? "text-xs font-medium text-neutral-100"
                  : "text-xs text-neutral-500 transition-colors hover:text-neutral-200"
              }
            >
              {item.label}
            </Link>
          ))}
        </nav>
        <a
          href="https://github.com/niloyrahman/edge-detect"
          target="_blank"
          rel="noreferrer"
          className="ml-auto text-xs text-neutral-500 transition-colors hover:text-neutral-200"
        >
          Source ↗
        </a>
      </div>
    </header>
  );
}
