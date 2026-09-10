import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/evidence")({
  head: () => ({
    meta: [
      { title: "Evidence — DeepShield-RAG" },
      {
        name: "description",
        content:
          "Evidence review for DeepShield-RAG.",
      },
      { property: "og:title", content: "Evidence — DeepShield-RAG" },
      {
        property: "og:description",
        content:
          "Evidence review for DeepShield-RAG.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: EvidencePage,
});

function EvidencePage() {
  return (
    <main className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
      <h1 className="font-display text-3xl font-bold tracking-tight text-foreground">
        Evidence
      </h1>
      <p className="mt-4 max-w-2xl text-muted-foreground">
        This page will host retrieved sources, model responses, and decision
        traces for review. It is currently a placeholder for the frontend
        shell.
      </p>
    </main>
  );
}
