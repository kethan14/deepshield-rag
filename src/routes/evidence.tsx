import { createFileRoute } from "@tanstack/react-router";
import { Archive } from "lucide-react";

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
} from "@/components/site-ui";

const title = "Evidence — DeepShield-RAG";
const description =
  "Interface preview of the DeepShield-RAG evidence log: filters for model, configuration, attack category, and outcome, with a reviewable record layout.";

export const Route = createFileRoute("/evidence")({
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
  component: EvidencePage,
});

const recordFields = [
  { label: "Prompt", note: "The request exactly as submitted" },
  { label: "Retrieved sources", note: "Passages with document provenance" },
  { label: "Guardrail decisions", note: "Which layer acted, and why" },
  { label: "Response", note: "Final answer returned to the user" },
  { label: "Evaluation", note: "Grounding and usefulness review" },
];

function EvidencePage() {
  return (
    <main>
      <PageIntro
        eyebrow="Workspace P-04"
        title="Evidence"
        description="Every evaluated request becomes one reviewable record: the prompt, what was retrieved, which guardrails acted, and what was returned."
        aside={
          <PrototypeNotice>
            Interface preview — verified experiment records will appear here once the evaluation
            backend is connected. Nothing is stored yet.
          </PrototypeNotice>
        }
      />

      <Shell className="py-12 lg:py-16">
        <Reveal>
          <Panel>
            <PanelHeader title="Filters" meta="0 matching records" />
            <div className="grid gap-4 p-5 sm:grid-cols-2 lg:grid-cols-4">
              <StaticSelect
                id="ev-model"
                label="Model"
                value="All models"
                options={["All models"]}
              />
              <StaticSelect
                id="ev-config"
                label="Configuration"
                value="All configurations"
                options={[
                  "All configurations",
                  "C0 — Basic RAG",
                  "C1 — Hardened RAG",
                  "C2 — Layered DeepShield",
                ]}
              />
              <StaticSelect
                id="ev-category"
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
              <StaticSelect
                id="ev-outcome"
                label="Outcome"
                value="All outcomes"
                options={[
                  "All outcomes",
                  "Blocked",
                  "Answered",
                  "Answered with caveat",
                  "Needs review",
                ]}
              />
            </div>
          </Panel>
        </Reveal>

        <div className="mt-6 grid gap-6 lg:grid-cols-12">
          <Reveal delay={0.08} className="lg:col-span-7">
            <Panel className="h-full">
              <PanelHeader title="Records" meta="Empty" />
              <EmptyState
                icon={<Archive className="h-5 w-5" aria-hidden="true" />}
                title="No evidence records"
                description="Verified experiment records will appear here after the evaluation backend is integrated. Records are written only from completed runs."
                className="py-20"
              />
            </Panel>
          </Reveal>

          <Reveal delay={0.14} className="lg:col-span-5">
            <Panel className="h-full">
              <PanelHeader title="Record structure" meta="Reference" />
              <ol className="divide-y divide-border">
                {recordFields.map((field, i) => (
                  <li key={field.label} className="flex gap-4 px-5 py-4">
                    <span className="label-mono w-8 shrink-0 pt-[2px]">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <span className="min-w-0">
                      <span className="block font-display text-[13.5px] font-medium text-foreground">
                        {field.label}
                      </span>
                      <span className="mt-1 block text-[12.5px] leading-relaxed text-muted-foreground">
                        {field.note}
                      </span>
                    </span>
                  </li>
                ))}
              </ol>
            </Panel>
          </Reveal>
        </div>
      </Shell>

      <Disclaimer />
    </main>
  );
}
