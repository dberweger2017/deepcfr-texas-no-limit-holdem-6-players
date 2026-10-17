//! Ordered gzip output, pieces formatted in parallel.
//! `Compatible`: one default-level stream, byte-identical. `Parallel`: one gzip member per piece.

use flate2::write::GzEncoder;
use flate2::Compression;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::mpsc::sync_channel;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Gzip {
    #[default]
    Compatible,
    Parallel,
}

/// `head`, then each piece's bytes in order, into `file`; returns it flushed, not synced.
pub fn write_ordered<P: Sync>(file: File, gzip: Gzip, head: &[u8], pieces: &[P],
                              format: impl Fn(&P, &mut Vec<u8>) + Sync) -> std::io::Result<File> {
    let mut file = BufWriter::new(file);
    let window = 2 * rayon::current_num_threads().max(1);
    let formatted = |piece: &P| {
        let mut bytes = Vec::new();
        format(piece, &mut bytes);
        bytes
    };
    match gzip {
        Gzip::Compatible => std::thread::scope(|scope| {
            let (send, receive) = sync_channel::<Vec<u8>>(window);
            let compressor = scope.spawn(move || -> std::io::Result<File> {
                let mut out = GzEncoder::new(file, Compression::default());
                for bytes in receive {
                    out.write_all(&bytes)?;
                }
                out.finish()?.into_inner().map_err(|e| e.into_error())
            });
            let mut open = send.send(head.to_vec()).is_ok();
            for window in pieces.chunks(window) {
                if !open {
                    break;
                }
                let texts: Vec<Vec<u8>> = window.par_iter().map(formatted).collect();
                open = texts.into_iter().all(|text| send.send(text).is_ok());
            }
            drop(send);
            compressor.join().expect("gzip compressor thread")
        }),
        Gzip::Parallel => {
            let member = |bytes: &[u8]| -> std::io::Result<Vec<u8>> {
                let mut out = GzEncoder::new(Vec::new(), Compression::default());
                out.write_all(bytes)?;
                out.finish()
            };
            file.write_all(&member(head)?)?;
            for window in pieces.chunks(window) {
                let members: Vec<std::io::Result<Vec<u8>>> = window.par_iter().map(|p| member(&formatted(p))).collect();
                for bytes in members {
                    file.write_all(&bytes?)?;
                }
            }
            file.into_inner().map_err(|e| e.into_error())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn both_modes_hold_the_same_text_and_compatible_matches_one_stream() {
        let dir = std::env::temp_dir().join(format!("gz-ordered-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pieces: Vec<u32> = (0..500).collect();
        let format = |p: &u32, out: &mut Vec<u8>| {
            for j in 0..(p % 37) {
                writeln!(out, "[{p},{j},\"row {}\"]", p * 1000 + j).unwrap();
            }
        };
        let mut text = b"{\"header\":1}\n".to_vec();
        pieces.iter().for_each(|p| format(p, &mut text));
        let mut single = GzEncoder::new(Vec::new(), Compression::default());
        single.write_all(&text).unwrap();
        let single = single.finish().unwrap();
        for gzip in [Gzip::Compatible, Gzip::Parallel] {
            let path = dir.join(format!("{gzip:?}.gz"));
            write_ordered(File::create(&path).unwrap(), gzip, b"{\"header\":1}\n", &pieces, format).unwrap();
            let bytes = std::fs::read(&path).unwrap();
            let mut decoded = Vec::new();
            flate2::read::MultiGzDecoder::new(&bytes[..]).read_to_end(&mut decoded).unwrap();
            assert_eq!(decoded, text, "{gzip:?}");
            assert_eq!(bytes == single, gzip == Gzip::Compatible, "{gzip:?}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
