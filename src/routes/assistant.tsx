import { createFileRoute } from "@tanstack/react-router";
import { FileText, MessagesSquare, SlidersHorizontal } from "lucide-react";

import { Reveal } from "@/components/motion-primitives";
import {
  Disclaimer,
  EmptyState,
  Panel,
  PanelHeader,
  PageIntro,
  PrototypeNotice,
  Shell,
  StaticSelect,
  Toggle,
} from "@/components/site-ui";

const title = "Policy Assistant — DeepShield-RAG";
const description =
  "Interface preview of the F-1 policy assistant workspace: conversation surface, retrieved-source panel, and guardrail configuration controls.";

export const Route = createFileRoute("/assistant")({
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
  component: AssistantPage,
});

function AssistantPage() {
  return (
    <main>
      <PageIntro
        eyebrow="Workspace P-01"
        title="Policy Assistant"
        description="A retrieval-augmented assistant for F-1 student policy questions, used as the demonstration subject for every evaluation in this project."
        aside={
          <PrototypeNotice>
            Interface preview — backend not connected. Conversations, retrieval, and guardrail
            execution are not yet wired to a running pipeline.
          </PrototypeNotice>
        }
      />

      <Shell className="grid gap-6 py-12 lg:grid-cols-12 lg:py-16">
        <Reveal className="lg:col-span-8">
          <Panel className="flex h-full min-h-[520px] flex-col">
            <PanelHeader title="Conversation" meta="Session not started">
              <span className="label-mono">C2 · Layered</span>
            </PanelHeader>

            <div className="flex flex-1 items-center justify-center">
              <EmptyState
                icon={<MessagesSquare className="h-5 w-5" aria-hidden="true" />}
                title="No conversation yet"
                description="Once the evaluation backend is connected, questions and their guarded responses will appear here in order."
              />
            </div>

            <div className="border-t border-border p-4">
              <label htmlFor="assistant-input" className="sr-only">
                Ask a policy question
              </label>
              <div className="rounded-[10px] border border-border bg-surface-alt p-3">
                <textarea
                  id="assistant-input"
                  rows={3}
                  disabled
                  placeholder="Ask a question about F-1 student policy…"
                  className="w-full resize-none bg-transparent text-[14px] text-foreground placeholder:text-muted-foreground focus:outline-none disabled:cursor-not-allowed"
                />
                <div className="mt-2 flex items-center justify-between border-t border-border pt-2">
                  <span className="label-mono">Input disabled in preview</span>
                  <span className="label-mono">Enter to send</span>
                </div>
              </div>
            </div>
          </Panel>
        </Reveal>

        <div className="flex flex-col gap-6 lg:col-span-4">
          <Reveal delay={0.08}>
            <Panel>
              <PanelHeader title="Retrieved sources" meta="0 passages" />
              <EmptyState
                icon={<FileText className="h-5 w-5" aria-hidden="true" />}
                title="No sources retrieved"
                description="Retrieved passages from public U.S. government documents will be listed here with their provenance."
                className="py-12"
              />
            </Panel>
          </Reveal>

          <Reveal delay={0.14}>
            <Panel>
              <PanelHeader title="Defense configuration">
                <SlidersHorizontal className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
              </PanelHeader>
              <div className="space-y-4 p-5">
                <StaticSelect
                  id="assistant-config"
                  label="Configuration"
                  value="C2 — Layered DeepShield"
                  options={["C0 — Basic RAG", "C1 — Hardened RAG", "C2 — Layered DeepShield"]}
                />
                <div className="divide-y divide-border border-t border-border">
                  <Toggle
                    id="guard-risk"
                    label="Risk analysis"
                    description="Classify the request before retrieval runs."
                    defaultChecked
                  />
                  <Toggle
                    id="guard-retrieval"
                    label="Retrieval inspection"
                    description="Examine retrieved passages for embedded instructions."
                    defaultChecked
                  />
                  <Toggle
                    id="guard-response"
                    label="Response evaluation"
                    description="Check grounding and usefulness before returning an answer."
                    defaultChecked
                  />
                </div>
              </div>
            </Panel>
          </Reveal>
        </div>
      </Shell>

      <Disclaimer />
    </main>
  );
}
