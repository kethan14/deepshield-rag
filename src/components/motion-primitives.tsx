import { motion, useReducedMotion, type Variants } from "framer-motion";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

const EASE = [0.22, 0.61, 0.36, 1] as const;

/** Gentle fade-and-rise reveal (16px), triggered once on scroll into view. */
export function Reveal({
  children,
  className,
  delay = 0,
  as = "div",
}: {
  children: ReactNode;
  className?: string;
  delay?: number;
  as?: "div" | "section" | "li" | "header" | "figure";
}) {
  const reduce = useReducedMotion();
  const Comp = motion[as];

  return (
    <Comp
      className={className}
      initial={reduce ? undefined : { opacity: 0, y: 16 }}
      whileInView={reduce ? undefined : { opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-64px" }}
      transition={{ duration: 0.6, ease: EASE, delay }}
    >
      {children}
    </Comp>
  );
}

/** Staggered container for lists of revealed children. */
export const staggerParent: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.09 } },
};

export const staggerChild: Variants = {
  hidden: { opacity: 0, y: 14 },
  show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: EASE } },
};

export function StaggerList({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  const reduce = useReducedMotion();
  if (reduce) return <ul className={className}>{children}</ul>;

  return (
    <motion.ul
      className={className}
      variants={staggerParent}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, margin: "-64px" }}
    >
      {children}
    </motion.ul>
  );
}

export function StaggerItem({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  const reduce = useReducedMotion();
  if (reduce) return <li className={className}>{children}</li>;

  return (
    <motion.li className={className} variants={staggerChild}>
      {children}
    </motion.li>
  );
}

/** Small hover lift, capped at 2px. */
export function HoverLift({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  const reduce = useReducedMotion();

  return (
    <motion.div
      className={cn(className)}
      whileHover={reduce ? undefined : { y: -2 }}
      transition={{ duration: 0.2, ease: EASE }}
    >
      {children}
    </motion.div>
  );
}

/** SVG path that draws itself when scrolled into view. */
export function DrawnPath({
  d,
  className,
  duration = 1.4,
  delay = 0,
  strokeWidth = 1,
  dashed = false,
}: {
  d: string;
  className?: string;
  duration?: number;
  delay?: number;
  strokeWidth?: number;
  dashed?: boolean;
}) {
  const reduce = useReducedMotion();

  return (
    <motion.path
      d={d}
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeDasharray={dashed ? "3 5" : undefined}
      initial={reduce || dashed ? undefined : { pathLength: 0 }}
      whileInView={reduce || dashed ? undefined : { pathLength: 1 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration, ease: "easeInOut", delay }}
    />
  );
}
