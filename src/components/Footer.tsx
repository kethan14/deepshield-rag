import { Link } from "@tanstack/react-router";

const columns = [
  {
    heading: "Workspaces",
    links: [
      { label: "Policy Assistant", to: "/assistant" as const },
      { label: "Attack Lab", to: "/attack-lab" as const },
    ],
  },
  {
    heading: "Research",
    links: [
      { label: "Experiments", to: "/experiments" as const },
      { label: "Evidence", to: "/evidence" as const },
    ],
  },
];

export function Footer() {
  return (
    <footer className="w-full border-t border-border bg-surface">
      <div className="mx-auto grid max-w-[1200px] gap-10 px-6 py-12 lg:grid-cols-12 lg:px-10">
        <div className="lg:col-span-5">
          <p className="font-display text-sm font-semibold tracking-tight text-foreground">
            DeepShield-RAG
          </p>
          <p className="mt-3 max-w-sm text-[13px] leading-relaxed text-muted-foreground">
            A security evaluation and guardrail benchmarking platform for
            retrieval-augmented AI assistants, developed as a master&apos;s
            capstone research project.
          </p>
        </div>

        {columns.map((col) => (
          <nav key={col.heading} className="lg:col-span-3" aria-label={col.heading}>
            <h2 className="label-mono">{col.heading}</h2>
            <ul className="mt-4 space-y-2.5">
              {col.links.map((link) => (
                <li key={link.to}>
                  <Link
                    to={link.to}
                    className="text-[13px] text-muted-foreground transition-colors hover:text-foreground"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
        ))}

        <div className="lg:col-span-1" />
      </div>

      <div className="border-t border-border">
        <div className="mx-auto flex max-w-[1200px] flex-col gap-2 px-6 py-6 sm:flex-row sm:items-center sm:justify-between lg:px-10">
          <p className="label-mono">Master&apos;s Capstone Research Project</p>
          <p className="text-[12px] text-muted-foreground">
            Interface prototype — no backend connected.
          </p>
        </div>
      </div>
    </footer>
  );
}
