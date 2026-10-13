// Where a photo was taken, from the GPS fields a camera or phone writes in a JPEG's EXIF.
// Read in the browser before the photo is shrunk (the shrunk copy keeps no EXIF). Phones often
// remove the position when a web page picks a photo; then there is none and nothing happens.

/** Bytes read from the start of the file: EXIF sits in the first APP1 segment. */
const HEAD_BYTES = 256 * 1024;

/** { latitude, longitude } written in the photo, or null. Never throws. */
export async function exifPosition(blob) {
  try {
    const view = new DataView(await blob.slice(0, HEAD_BYTES).arrayBuffer());
    if (view.getUint16(0) !== 0xffd8) return null; // not a JPEG
    let offset = 2;
    while (offset + 10 <= view.byteLength) {
      const marker = view.getUint16(offset);
      if ((marker & 0xff00) !== 0xff00 || marker === 0xffda) return null; // image data: no EXIF
      const size = view.getUint16(offset + 2);
      if (marker === 0xffe1 && view.getUint32(offset + 4) === 0x45786966) { // "Exif"
        return gpsFromTiff(view, offset + 10);
      }
      offset += 2 + size;
    }
  } catch {
    // A truncated or odd file: no position.
  }
  return null;
}

function gpsFromTiff(view, tiff) {
  const little = view.getUint16(tiff) === 0x4949; // "II": Intel byte order
  const u16 = (at) => view.getUint16(tiff + at, little);
  const u32 = (at) => view.getUint32(tiff + at, little);
  const entries = (ifd) => Array.from({ length: u16(ifd) }, (_, i) => ifd + 2 + i * 12);
  const pointer = entries(u32(4)).find((e) => u16(e) === 0x8825); // GPS IFD
  if (pointer === undefined) return null;
  const tags = new Map(entries(u32(pointer + 8)).map((e) => [u16(e), e]));
  const letter = (tag) => (tags.has(tag) ? String.fromCharCode(view.getUint8(tiff + tags.get(tag) + 8)) : '');
  const degrees = (tag) => {
    if (!tags.has(tag)) return null;
    const at = u32(tags.get(tag) + 8); // three RATIONALs: degrees, minutes, seconds
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
  if (letter(1) === 'S') latitude = -latitude;
  if (letter(3) === 'W') longitude = -longitude;
  const blank = latitude === 0 && longitude === 0;
  if (blank || Math.abs(latitude) > 90 || Math.abs(longitude) > 180) return null;
  return { latitude, longitude };
}
