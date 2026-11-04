// The visitor's photo in the browser: kept in a canvas, cut and shrunk before it is sent.
// Parts and outlines are cut out here, so the server always gets one plain photo.

import type { Position } from "./api";

/** Longest side kept for display and for cutting a part. */
const KEEP_SIDE = 2048;
/** Longest side of a photo sent to search (the server shrinks it to 512 px again). */
export const SEND_SIDE = 1024;
/** Longest side of the photo sent to find its objects (`DETECT_SIDE`). */
export const DETECT_SIDE = 1024;

/** A part of the photo, in shares of its width and height (0 to 1). */
export type Part = { x: number; y: number; w: number; h: number; label?: string };

export type Photo = {
  canvas: HTMLCanvasElement;
  url: string;
  name: string;
  /** Where the photo was taken (its EXIF GPS), when it says. */
  position: Position | null;
  /** Credit line of an example photo. */
  credit?: string;
};

export async function loadPhoto(blob: Blob, name: string, credit?: string): Promise<Photo> {
  const position = await exifPosition(blob);
  const bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
  const scale = Math.min(1, KEEP_SIDE / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d")!.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  const url = URL.createObjectURL(await toBlob(canvas, 0.92));
  return { canvas, url, name, position, credit };
}

function toBlob(canvas: HTMLCanvasElement, quality: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("encode"))), "image/jpeg", quality);
  });
}

function dataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.readAsDataURL(blob);
  });
}

/** The photo, or a part of it, as a JPEG data URL at most `side` on its long side. */
export async function cut(photo: Photo, part: Part | null, side: number): Promise<string> {
  const source = photo.canvas;
  const box = part
    ? { x: part.x * source.width, y: part.y * source.height, w: part.w * source.width, h: part.h * source.height }
    : { x: 0, y: 0, w: source.width, h: source.height };
  const scale = Math.min(1, side / Math.max(box.w, box.h));
  const out = document.createElement("canvas");
  out.width = Math.max(1, Math.round(box.w * scale));
  out.height = Math.max(1, Math.round(box.h * scale));
  out.getContext("2d")!.drawImage(source, box.x, box.y, box.w, box.h, 0, 0, out.width, out.height);
  return dataUrl(await toBlob(out, 0.9));
}

// --- where the photo was taken ---------------------------------------------------------------
// Read before the photo is shrunk (the shrunk copy keeps no EXIF). Phones often remove the
// position when a web page picks a photo; then there is none and nothing happens.

/** Bytes read from the start of the file: EXIF sits in the first APP1 segment. */
const HEAD_BYTES = 256 * 1024;

/** The position written in the photo, or null. Never throws. */
export async function exifPosition(blob: Blob): Promise<Position | null> {
  try {
    const view = new DataView(await blob.slice(0, HEAD_BYTES).arrayBuffer());
    if (view.getUint16(0) !== 0xffd8) return null; // not a JPEG
    let offset = 2;
    while (offset + 10 <= view.byteLength) {
      const marker = view.getUint16(offset);
      if ((marker & 0xff00) !== 0xff00 || marker === 0xffda) return null; // image data: no EXIF
      const size = view.getUint16(offset + 2);
      if (marker === 0xffe1 && view.getUint32(offset + 4) === 0x45786966) return gpsFromTiff(view, offset + 10); // "Exif"
      offset += 2 + size;
    }
  } catch {
    // A truncated or odd file: no position.
  }
  return null;
}

function gpsFromTiff(view: DataView, tiff: number): Position | null {
  const little = view.getUint16(tiff) === 0x4949; // "II": Intel byte order
  const u16 = (at: number) => view.getUint16(tiff + at, little);
  const u32 = (at: number) => view.getUint32(tiff + at, little);
  const entries = (ifd: number) => Array.from({ length: u16(ifd) }, (_, i) => ifd + 2 + i * 12);
  const pointer = entries(u32(4)).find((e) => u16(e) === 0x8825); // GPS IFD
  if (pointer === undefined) return null;
  const tags = new Map(entries(u32(pointer + 8)).map((e) => [u16(e), e]));
  const letter = (tag: number) => (tags.has(tag) ? String.fromCharCode(view.getUint8(tiff + tags.get(tag)! + 8)) : "");
  const degrees = (tag: number) => {
    if (!tags.has(tag)) return null;
    const at = u32(tags.get(tag)! + 8); // three RATIONALs: degrees, minutes, seconds
    let value = 0;
    for (let i = 0; i < 3; i += 1) {
      const den = u32(at + i * 8 + 4);
      value += den ? u32(at + i * 8) / den / 60 ** i : 0;
    }
    return value;
  };
  let latitude = degrees(2);
  let longitude = degrees(4);
  if (latitude === null || longitude === null) return null;
  if (letter(1) === "S") latitude = -latitude;
  if (letter(3) === "W") longitude = -longitude;
  const blank = latitude === 0 && longitude === 0;
  if (blank || Math.abs(latitude) > 90 || Math.abs(longitude) > 180) return null;
  return { latitude, longitude };
}

/** Where the browser says the visitor stands, or null when refused or unavailable. */
export function devicePosition(): Promise<Position | null> {
  return new Promise((resolve) => {
    if (!("geolocation" in navigator)) {
      resolve(null);
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (found) => resolve({ latitude: found.coords.latitude, longitude: found.coords.longitude }),
      () => resolve(null),
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 },
    );
  });
}
