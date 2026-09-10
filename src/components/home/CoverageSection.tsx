import { Reveal } from "@/components/motion-primitives";
import { SectionHeading, Shell } from "@/components/site-ui";

const configs = ["C0", "C1", "C2"] as const;

const categories = [
  {
    id: "T-01",
    name: "Direct Injection",
    scope: "Instructions placed in the user request that contradict the assigned policy.",
  },
  {
    id: "T-02",
    name: "Indirect Injection",
    scope: "Instructions embedded in retrieved documents rather than in the request itself.",
  },
  {
    id: "T-03",
    name: "Prompt Extraction",
    scope: "Attempts to recover system instructions or configuration text.",
  },
  {
    id: "T-04",
    name: "Data Leakage",
    scope: "Disclosure of restricted context that the question did not warrant.",
  },
  {
    id: "T-05",
    name: "Citation Manipulation",
    scope: "Answers attributed to sources that do not support the claim made.",
  },
  {
    id: "T-06",
    name: "Unauthorized Tool Use",
    scope: "Requests that push the assistant beyond its permitted actions.",
  },
];

export function CoverageSection() {
  return (
    <section className="border-b border-border" aria-labelledby="coverage-heading">
      <Shell className="py-16 lg:py-24">
        <div className="flex flex-wrap items-end justify-between gap-6">
          <SectionHeading
            index="04"
            eyebrow="Benchmark coverage"
            title="Six attack categories form the benchmark."
            description="Each category will be exercised against every configuration. The matrix below records intended coverage only — no category has a measured outcome yet."
          />
          <Reveal delay={0.1}>
            <p className="label-mono rounded-[10px] border border-dashed border-[color:var(--line)] px-3 py-2">
              Planned coverage — no results recorded
            </p>
          </Reveal>
        </div>

        <Reveal delay={0.08} className="mt-12 overflow-x-auto">
          <table className="w-full min-w-[680px] border-collapse text-left">
            <caption className="sr-only">
              Planned benchmark coverage of six attack categories across three
              experimental configurations.
            </caption>
            <thead>
              <tr className="border-y border-border">
                <th scope="col" className="label-mono py-3 pr-4 font-normal">
                  Ref
                </th>
                <th scope="col" className="label-mono py-3 pr-4 font-normal">
                  Category
                </th>
                <th scope="col" className="label-mono py-3 pr-4 font-normal">
                  Scope
                </th>
                {configs.map((c) => (
                  <th
                    key={c}
                    scope="col"
                    className="label-mono w-[72px] py-3 text-center font-normal"
                  >
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {categories.map((cat) => (
                <tr
                  key={cat.id}
                  className="border-b border-border align-top transition-colors hover:bg-surface-alt"
                >
                  <td className="py-4 pr-4 font-mono text-[12px] text-muted-foreground">
                    {cat.id}
                  </td>
                  <td className="py-4 pr-4">
                    <span className="font-display text-[14px] font-medium text-foreground">
                      {cat.name}
                    </span>
                  </td>
                  <td className="max-w-md py-4 pr-4 text-[13px] leading-relaxed text-muted-foreground">
                    {cat.scope}
                  </td>
                  {configs.map((c) => (
                    <td key={c} className="py-4 text-center">
                      <span
                        className="mx-auto block h-[9px] w-[9px] rounded-[2px] border border-[color:var(--line)]"
                        aria-hidden="true"
                      />
                      <span className="sr-only">Planned</span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </Reveal>

        <Reveal delay={0.12} className="mt-6 flex items-center gap-2">
          <span
            className="h-[9px] w-[9px] rounded-[2px] border border-[color:var(--line)]"
            aria-hidden="true"
          />
          <span className="label-mono">
            Planned evaluation cell — outcome not yet recorded
          </span>
        </Reveal>
      </Shell>
    </section>
  );
}
