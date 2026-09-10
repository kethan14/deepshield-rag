import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/experiments")({
  head: () => ({
    meta: [
      { title: "Experiments — DeepShield-RAG" },
      {
        name: "description",
        content:
          "Experiment management for DeepShield-RAG.",
      },
      { property: "og:title", content: "Experiments — DeepShield-RAG" },
      {
        property: "og:description",
        content:
          "Experiment management for DeepShield-RAG.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: ExperimentsPage,
});

function ExperimentsPage() {
  return (
    <main className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
      <h1 className="font-display text-3xl font-bold tracking-tight text-foreground">
        Experiments
      </h1>
      <p className="mt-4 max-w-2xl text-muted-foreground">
        This page will host experiment configuration and results. It is
        currently a placeholder for the frontend shell.
      </p>
    </main>
  );
}
