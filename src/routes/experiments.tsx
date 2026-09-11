import { createFileRoute } from "@tanstack/react-router";
import { BarChart3, Table2 } from "lucide-react";

import { Reveal } from "@/components/motion-primitives";
import {
  Disclaimer,
  EmptyState,
  PageIntro,
  Panel,
  PanelHeader,
  PrototypeNotice,
  Shell,
  StaticSelect,
  Toggle,
} from "@/components/site-ui";

const title = "Experiments — DeepShield-RAG";
const description =
  "Planned experimental design for DeepShield-RAG: configuration comparison controls and reserved surfaces for charts and comparison tables.";

export const Route = createFileRoute("/experiments")({
  head: () => ({
    meta: [
      { title },
      { name: "description", content: description },
      { property: "og:title", content: title },
      { property: "og:description", content: description },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: ExperimentsPage,
});

const configurations = [
  {
    id: "C0",
    name: "Basic RAG",
    note: "Reference path with no dedicated security layer.",
  },
  {
    id: "C1",
    name: "Hardened RAG",
    note: "Prompt hardening and basic output constraints.",
  },
  {
    id: "C2",
    name: "Layered DeepShield",
    note: "Full risk analysis, inspection, guardrails, and evidence capture.",
  },
];

function ExperimentsPage() {
  return (
    <main>
      <PageIntro
        eyebrow="Workspace P-03"
        title="Experiments"
        description="The controlled comparison between configurations. Documents, model, and question set are held constant; only the defensive layers change."
        aside={
          <PrototypeNotice>
            Planned experimental design. Comparison results will appear here after the evaluation
            backend is integrated; no measurements exist yet.
          </PrototypeNotice>
        }
      />

      <Shell className="grid gap-6 py-12 lg:grid-cols-12 lg:py-16">
        <Reveal className="lg:col-span-4">
          <Panel>
            <PanelHeader title="Comparison controls" />
            <div className="space-y-5 p-5">
              <StaticSelect
                id="exp-baseline"
                label="Baseline"
                value="C0 — Basic RAG"
                options={["C0 — Basic RAG", "C1 — Hardened RAG", "C2 — Layered DeepShield"]}
              />
              <StaticSelect
                id="exp-candidate"
                label="Candidate"
                value="C2 — Layered DeepShield"
                options={["C0 — Basic RAG", "C1 — Hardened RAG", "C2 — Layered DeepShield"]}
              />
              <StaticSelect
                id="exp-category"
                label="Attack category"
                value="All categories"
                options={[
                  "All categories",
                  "Direct Injection",
                  "Indirect Injection",
                  "Prompt Extraction",
                  "Data Leakage",
                  "Citation Manipulation",
                  "Unauthorized Tool Use",
                ]}
              />
              <div className="divide-y divide-border border-t border-border">
                <Toggle
                  id="exp-usefulness"
                  label="Include usefulness review"
                  description="Report answer quality alongside security outcomes."
                  defaultChecked
                />
                <Toggle
                  id="exp-repeat"
                  label="Repeat each prompt"
                  description="Run every prompt more than once to observe variance."
                />
              </div>
            </div>
          </Panel>
        </Reveal>

        <div className="flex flex-col gap-6 lg:col-span-8">
          <Reveal delay={0.08}>
            <Panel>
              <PanelHeader title="Planned configurations" meta="Not yet executed" />
              <ul className="divide-y divide-border">
                {configurations.map((config) => (
                  <li
                    key={config.id}
                    className="flex flex-wrap items-center gap-x-4 gap-y-1 px-5 py-4"
                  >
                    <span className="font-mono text-[13px] text-primary">{config.id}</span>
                    <span className="font-display text-[14px] font-medium text-foreground">
                      {config.name}
                    </span>
                    <span className="min-w-0 flex-1 text-[13px] text-muted-foreground">
                      {config.note}
                    </span>
                    <span className="label-mono">Planned</span>
                  </li>
                ))}
              </ul>
            </Panel>
          </Reveal>

          <Reveal delay={0.14}>
            <Panel>
              <PanelHeader title="Comparison chart" meta="Reserved" />
              <div className="p-5">
                <div className="grid-field h-[220px] rounded-[10px] border border-dashed border-[color:var(--line)]" />
                <p className="label-mono mt-4">
                  Reserved for security and usefulness comparison across configurations
                </p>
              </div>
            </Panel>
          </Reveal>

          <Reveal delay={0.18} className="grid gap-6 sm:grid-cols-2">
            <Panel>
              <PanelHeader title="Comparison table" meta="Reserved" />
              <EmptyState
                icon={<Table2 className="h-5 w-5" aria-hidden="true" />}
                title="No comparison data"
                description="Per-category outcomes will populate this table once runs are recorded."
                className="py-12"
              />
            </Panel>
            <Panel>
              <PanelHeader title="Variance view" meta="Reserved" />
              <EmptyState
                icon={<BarChart3 className="h-5 w-5" aria-hidden="true" />}
                title="No repeated runs"
                description="Repeat-run distributions will be summarised here after integration."
                className="py-12"
              />
            </Panel>
          </Reveal>
        </div>
      </Shell>

      <Disclaimer />
    </main>
  );
}
