import { Fragment, type ReactNode } from "react";

/* The model answers in markdown — headings, bold, bullets, fenced code. This
 * renders the subset it actually emits. Everything becomes React elements, so
 * there is no dangerouslySetInnerHTML and no HTML injection from model output.
 */

const INLINE = /(`[^`]+`|\*\*[^*]+\*\*)/g;

function inline(text: string, key: string): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    if (part.startsWith("`") && part.endsWith("`") && part.length > 1)
      return <code key={`${key}-${i}`}>{part.slice(1, -1)}</code>;
    // Bold often wraps a path in backticks (**`src/flask/app.py`**), so recurse
    // rather than dropping the inner span as literal text.
    if (part.startsWith("**") && part.endsWith("**") && part.length > 3)
      return <strong key={`${key}-${i}`}>{inline(part.slice(2, -2), `${key}-${i}b`)}</strong>;
    return <Fragment key={`${key}-${i}`}>{part}</Fragment>;
  });
}

const BULLET = /^\s*[-*]\s+/;
const NUMBERED = /^\s*\d+\.\s+/;
const HEADING = /^(#{1,6})\s+(.*)$/;

/** One non-code segment: group consecutive lines into headings, lists, paragraphs. */
function blocks(src: string, key: string): ReactNode[] {
  const out: ReactNode[] = [];
  const lines = src.split("\n");
  let para: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;

  const flushPara = () => {
    if (!para.length) return;
    out.push(<p key={`${key}-p${out.length}`}>{inline(para.join(" "), `${key}-${out.length}`)}</p>);
    para = [];
  };
  const flushList = () => {
    if (!list) return;
    const items = list.items.map((t, i) => <li key={i}>{inline(t, `${key}-li${i}`)}</li>);
    out.push(
      list.ordered ? (
        <ol key={`${key}-l${out.length}`}>{items}</ol>
      ) : (
        <ul key={`${key}-l${out.length}`}>{items}</ul>
      )
    );
    list = null;
  };

  for (const line of lines) {
    const heading = line.match(HEADING);
    if (heading) {
      flushPara();
      flushList();
      // Levels collapse to two ranks — the answers only ever nest that deep.
      const Tag = heading[1].length <= 2 ? "h3" : "h4";
      out.push(<Tag key={`${key}-h${out.length}`}>{inline(heading[2], `${key}-${out.length}`)}</Tag>);
      continue;
    }
    const ordered = NUMBERED.test(line);
    if (ordered || BULLET.test(line)) {
      flushPara();
      if (!list || list.ordered !== ordered) {
        flushList();
        list = { ordered, items: [] };
      }
      list.items.push(line.replace(ordered ? NUMBERED : BULLET, ""));
      continue;
    }
    if (!line.trim()) {
      flushPara();
      flushList();
      continue;
    }
    flushList();
    para.push(line.trim());
  }
  flushPara();
  flushList();
  return out;
}

export function Markdown({ text }: { text: string }) {
  // Split on fences first so nothing inside a code block gets parsed as markdown.
  return (
    <div className="answer">
      {text.split(/```/).map((seg, i) =>
        i % 2 ? (
          // Odd segments are fenced code; drop the language tag on the first line.
          <pre key={i}>
            <code>{seg.replace(/^[a-zA-Z]*\n/, "").replace(/\n$/, "")}</code>
          </pre>
        ) : (
          <Fragment key={i}>{blocks(seg, `s${i}`)}</Fragment>
        )
      )}
    </div>
  );
}
