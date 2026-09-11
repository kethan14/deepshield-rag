import { Link } from "@tanstack/react-router";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";
import { Reveal } from "@/components/motion-primitives";

export type RouteTo = "/" | "/assistant" | "/attack-lab" | "/experiments" | "/evidence";

export function Shell({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn("mx-auto max-w-[1200px] px-6 lg:px-10", className)}>{children}</div>;
}

export function Eyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cn("label-mono", className)}>{children}</p>;
}

export function SectionHeading({
  index,
  eyebrow,
  title,
  description,
  className,
}: {
  index?: string;
  eyebrow?: string;
  title: string;
  description?: string;
  className?: string;
}) {
  return (
    <Reveal className={cn("max-w-2xl", className)}>
      {(index || eyebrow) && (
        <div className="flex items-center gap-3">
          {index && <span className="label-mono text-foreground/70">{index}</span>}
          {index && eyebrow && <span className="h-[1px] w-8 bg-border" aria-hidden="true" />}
          {eyebrow && <Eyebrow>{eyebrow}</Eyebrow>}
        </div>
      )}
      <h2 className="mt-4 font-display text-[26px] font-semibold leading-[1.2] text-foreground sm:text-[32px]">
        {title}
      </h2>
      {description && (
        <p className="mt-4 text-[15px] leading-relaxed text-muted-foreground">{description}</p>
      )}
    </Reveal>
  );
}

export function ActionLink({
  to,
  children,
  variant = "primary",
  className,
}: {
  to: RouteTo;
  children: ReactNode;
  variant?: "primary" | "secondary";
  className?: string;
}) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-[10px] px-5 py-2.5 text-[13px] font-medium transition-all duration-200";

  return (
    <Link
      to={to}
      className={cn(
        base,
        variant === "primary"
          ? "bg-primary text-primary-foreground hover:bg-foreground"
          : "border border-border bg-surface text-foreground hover:border-[color:var(--line)] hover:bg-surface-alt",
        className,
      )}
    >
      {children}
    </Link>
  );
}

export function Panel({
  children,
  className,
  as: Comp = "div",
}: {
  children: ReactNode;
  className?: string;
  as?: "div" | "section" | "article" | "aside";
}) {
  return (
    <Comp
      className={cn(
        "rounded-[14px] border border-border bg-surface",
        "shadow-[0_1px_2px_rgba(16,42,67,0.04),0_8px_24px_-16px_rgba(16,42,67,0.16)]",
        className,
      )}
    >
      {children}
    </Comp>
  );
}

export function PanelHeader({
  title,
  meta,
  children,
}: {
  title: string;
  meta?: string;
  children?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-3.5">
      <div className="flex min-w-0 items-center gap-3">
        <h3 className="truncate font-display text-[13px] font-semibold tracking-tight text-foreground">
          {title}
        </h3>
        {meta && <span className="label-mono shrink-0">{meta}</span>}
      </div>
      {children}
    </div>
  );
}

/** Honest state label: this interface has no backend behind it. */
export function PrototypeNotice({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-[10px] border border-dashed border-[color:var(--line)] bg-surface-alt px-4 py-3",
        className,
      )}
    >
      <span
        className="mt-[7px] h-[6px] w-[6px] shrink-0 rounded-full bg-[color:var(--line)]"
        aria-hidden="true"
      />
      <p className="text-[12.5px] leading-relaxed text-muted-foreground">{children}</p>
    </div>
  );
}

export function EmptyState({
  title,
  description,
  icon,
  className,
}: {
  title: string;
  description: string;
  icon?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn("flex flex-col items-center justify-center px-8 py-16 text-center", className)}
    >
      {icon && (
        <span className="mb-5 grid h-11 w-11 place-items-center rounded-[10px] border border-border bg-surface-alt text-muted-foreground">
          {icon}
        </span>
      )}
      <h3 className="font-display text-[15px] font-semibold text-foreground">{title}</h3>
      <p className="mt-2 max-w-sm text-[13px] leading-relaxed text-muted-foreground">
        {description}
      </p>
    </div>
  );
}

export function PageIntro({
  eyebrow,
  title,
  description,
  aside,
}: {
  eyebrow: string;
  title: string;
  description: string;
  aside?: ReactNode;
}) {
  return (
    <div className="border-b border-border bg-surface-alt">
      <Shell className="py-12 lg:py-16">
        <div className="grid gap-8 lg:grid-cols-12 lg:items-end">
          <div className="lg:col-span-7">
            <Eyebrow>{eyebrow}</Eyebrow>
            <h1 className="mt-4 font-display text-[30px] font-semibold leading-[1.15] tracking-tight text-foreground sm:text-[38px]">
              {title}
            </h1>
            <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-muted-foreground">
              {description}
            </p>
          </div>
          {aside && <div className="lg:col-span-5">{aside}</div>}
        </div>
      </Shell>
    </div>
  );
}

/** Non-interactive control surfaces used across the preview workspaces. */
export function StaticSelect({
  label,
  value,
  options,
  id,
}: {
  label: string;
  value: string;
  options: string[];
  id: string;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={id} className="label-mono">
        {label}
      </label>
      <select
        id={id}
        defaultValue={value}
        className="w-full rounded-[10px] border border-border bg-surface px-3 py-2 text-[13px] text-foreground transition-colors hover:border-[color:var(--line)]"
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  );
}

export function Toggle({
  label,
  description,
  defaultChecked = false,
  id,
}: {
  label: string;
  description: string;
  defaultChecked?: boolean;
  id: string;
}) {
  return (
    <div className="flex items-start gap-3 py-3">
      <input
        id={id}
        type="checkbox"
        defaultChecked={defaultChecked}
        className="mt-[3px] h-4 w-4 shrink-0 rounded-[4px] border border-border accent-[color:var(--primary)]"
      />
      <div className="min-w-0">
        <label htmlFor={id} className="block font-display text-[13px] font-medium text-foreground">
          {label}
        </label>
        <p className="mt-1 text-[12.5px] leading-relaxed text-muted-foreground">{description}</p>
      </div>
    </div>
  );
}

export function Disclaimer() {
  return (
    <div className="border-t border-border bg-surface-alt">
      <Shell className="py-8">
        <p className="mx-auto max-w-3xl text-center text-[12.5px] leading-relaxed text-muted-foreground">
          The F-1 policy assistant is an academic research demonstration based on public U.S.
          government documents. It does not provide legal advice.
        </p>
      </Shell>
    </div>
  );
}
