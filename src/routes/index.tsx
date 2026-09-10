import { createFileRoute } from "@tanstack/react-router";

import { IntroSection } from "@/components/home/IntroSection";
import { FailureSection } from "@/components/home/FailureSection";
import { ProcessSection } from "@/components/home/ProcessSection";
import { ComparisonSection } from "@/components/home/ComparisonSection";
import { CoverageSection } from "@/components/home/CoverageSection";
import { ClosingSection } from "@/components/home/ClosingSection";
import { Disclaimer } from "@/components/site-ui";

const title = "DeepShield-RAG — Security Evaluation for RAG Assistants";
const description =
  "DeepShield-RAG is a master's capstone research platform that benchmarks adversarial attacks, layered guardrails, and evidence capture for retrieval-augmented AI assistants.";

export const Route = createFileRoute("/")({
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
  component: Index,
});

function Index() {
  return (
    <main>
      <IntroSection />
      <FailureSection />
      <ProcessSection />
      <ComparisonSection />
      <CoverageSection />
      <ClosingSection />
      <Disclaimer />
    </main>
  );
}
