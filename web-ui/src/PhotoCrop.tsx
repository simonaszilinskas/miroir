import { useRef, useState, type PointerEvent } from "react";
import type { Part } from "./photo";

type Props = {
  src: string;
  alt: string;
  crop: Part | null;
  onCrop: (crop: Part | null) => void;
  /** Works the server found in the photo (`/api/detect`), biggest and most central first. */
  outlines: Part[];
};

/** Smallest frame, as a share of the photo, that counts as drawn (a smaller move is a tap). */
const MIN_SHARE = 0.03;

const toStyle = (b: Part) => ({ left: `${b.x * 100}%`, top: `${b.y * 100}%`, width: `${b.w * 100}%`, height: `${b.h * 100}%` });
const same = (a: Part | null, b: Part) => a !== null && a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h;

/** The visitor's photo, with numbered outlines to tap, or a frame to draw by dragging. */
export function PhotoCrop({ src, alt, crop, onCrop, outlines }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const start = useRef<{ x: number; y: number } | null>(null);
  const [drawing, setDrawing] = useState(false);

  const share = (e: PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)),
      y: Math.max(0, Math.min(1, (e.clientY - r.top) / r.height)),
    };
  };

  const down = (e: PointerEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).closest(".mr-box")) return;
    start.current = share(e);
    ref.current!.setPointerCapture(e.pointerId);
  };
  const move = (e: PointerEvent<HTMLDivElement>) => {
    if (!start.current) return;
    const p = share(e);
    const s = start.current;
    const c = { x: Math.min(s.x, p.x), y: Math.min(s.y, p.y), w: Math.abs(p.x - s.x), h: Math.abs(p.y - s.y) };
    if (c.w > MIN_SHARE && c.h > MIN_SHARE) {
      setDrawing(true);
      onCrop(c);
    }
  };
  const up = () => {
    start.current = null;
    setDrawing(false);
  };

  const chosen = outlines.findIndex((b) => same(crop, b));
  const drawn = crop !== null && chosen === -1;

  return (
    <div
      ref={ref}
      className={`mr-photo${crop ? " has-crop" : ""}${drawing ? " is-drawing" : ""}`}
      onPointerDown={down}
      onPointerMove={move}
      onPointerUp={up}
      onPointerCancel={up}
    >
      <img src={src} alt={alt} draggable={false} />
      {drawn && <div className="mr-frame" style={toStyle(crop)} aria-hidden="true" />}
      {!drawn &&
        outlines.map((b, i) =>
          crop && chosen !== i ? null : (
            <button
              key={`${b.x}-${b.y}-${b.w}-${b.h}`}
              type="button"
              className={`mr-box${chosen === i ? " is-on" : ""}`}
              style={{ ...toStyle(b), animationDelay: `${100 + Math.min(i, 12) * 60}ms` }}
              aria-pressed={chosen === i}
              aria-label={`Viser l’œuvre ${i + 1}`}
              onClick={() => onCrop(chosen === i ? null : b)}
            >
              <span aria-hidden="true">{i + 1}</span>
            </button>
          ),
        )}
    </div>
  );
}
