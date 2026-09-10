import { createFileRoute, Link } from "@tanstack/react-router";
import { FlaskConical, Scale, Search, ShieldCheck } from "lucide-react";
import { Card, CardDescription, CardTitle } from "@/components/ui/card";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      {
        title: "DeepShield-RAG — Security Evaluation for RAG Assistants",
      },
      {
        name: "description",
        content:
          "DeepShield-RAG measures the security and usefulness of retrieval-augmented generation assistants through structured attack testing, defense comparison, and evidence analysis.",
      },
      {
        property: "og:title",
        content: "DeepShield-RAG — Security Evaluation for RAG Assistants",
      },
      {
        property: "og:description",
        content:
          "DeepShield-RAG measures the security and usefulness of retrieval-augmented generation assistants through structured attack testing, defense comparison, and evidence analysis.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Index() {
  return (
    <main className="flex min-h-[calc(100vh-4rem-8rem)] flex-col">
      <section className="relative overflow-hidden px-4 pb-16 pt-20 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-4xl text-center">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
            <FlaskConical className="h-3.5 w-3.5" aria-hidden="true" />
            Research Prototype
          </div>
          <h1 className="font-display text-4xl font-bold tracking-tight text-foreground sm:text-5xl lg:text-6xl">
            Measure the security and usefulness of RAG assistants
          </h1>
          <p className="mx-auto mt-6 max-w-2xl text-lg text-muted-foreground">
            DeepShield-RAG is a security evaluation and guardrail benchmarking
            platform for retrieval-augmented AI assistants. Its demonstration
            use case is an F-1 student policy assistant based on public U.S.
            government documents.
          </p>
          <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
            <Link
              to="/policy-assistant"
              className="inline-flex items-center justify-center rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
            >
              Open Policy Assistant
            </Link>
            <Link
              to="/attack-lab"
              className="inline-flex items-center justify-center rounded-lg border border-border bg-secondary px-5 py-2.5 text-sm font-medium text-secondary-foreground transition-colors hover:bg-secondary/80"
            >
              Explore Attack Lab
            </Link>
          </div>
        </div>
      </section>

      <section className="px-4 py-16 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            <FeatureCard
              icon={<Search className="h-6 w-6 text-primary" />}
              title="Test Attacks"
              description="Probe retrieval-augmented assistants with structured adversarial prompts to surface failure modes and boundary violations."
            />
            <FeatureCard
              icon={<ShieldCheck className="h-6 w-6 text-primary" />}
              title="Compare Defenses"
              description="Evaluate guardrail configurations side by side to understand the trade-offs between security and helpfulness."
            />
            <FeatureCard
              icon={<Scale className="h-6 w-6 text-primary" />}
              title="Analyze Evidence"
              description="Review retrieved sources, model responses, and decision traces to build reproducible evidence for each finding."
            />
          </div>
        </div>
      </section>

      <section className="mt-auto px-4 pb-8 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-3xl rounded-lg border border-border/60 bg-secondary/50 p-4 text-center">
          <p className="text-sm text-muted-foreground">
            The F-1 student policy assistant is an academic demonstration built
            on public U.S. government documents. It does not provide legal
            advice.
          </p>
        </div>
      </section>
    </main>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <Card className="flex flex-col">
      <div className="mb-4 inline-flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
        {icon}
      </div>
      <CardTitle>{title}</CardTitle>
      <CardDescription className="flex-grow">{description}</CardDescription>
    </Card>
  );
}
