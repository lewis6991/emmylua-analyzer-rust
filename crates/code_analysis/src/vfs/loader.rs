use encoding_rs::{Encoding, UTF_8};
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};
use wax::Pattern;

use log::error;
use walkdir::WalkDir;

#[derive(Debug)]
pub struct LuaFileInfo {
    pub path: String,
    pub content: String,
}

impl LuaFileInfo {
    pub fn into_tuple(self) -> (PathBuf, Option<String>) {
        (PathBuf::from(self.path), Some(self.content))
    }
}

#[allow(unused)]
pub fn load_workspace_files(
    root: &Path,
    include_pattern: &Vec<String>,
    exclude_pattern: &Vec<String>,
    exclude_dir: &Vec<PathBuf>,
    encoding: Option<&str>,
) -> Result<Vec<LuaFileInfo>, Box<dyn Error>> {
    let encoding = encoding.unwrap_or("utf-8");
    let mut files = Vec::new();
    let include_pattern = include_pattern
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<&str>>();

    let include_set = match wax::any(include_pattern) {
        Ok(glob) => glob,
        Err(e) => {
            error!("Invalid glob pattern: {:?}", e);
            return Ok(files);
        }
    };

    let exclude_pattern = exclude_pattern
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<&str>>();
    let exclude_set = match wax::any(exclude_pattern) {
        Ok(glob) => glob,
        Err(e) => {
            error!("Invalid ignore glob pattern: {:?}", e);
            return Ok(files);
        }
    };

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        if exclude_dir.iter().any(|dir| path.starts_with(dir)) {
            continue;
        }

        let relative_path = path.strip_prefix(root).unwrap();
        if exclude_set.is_match(relative_path) {
            continue;
        }

        if include_set.is_match(relative_path) {
            if let Some(content) = read_file_with_encoding(path, encoding) {
                files.push(LuaFileInfo {
                    path: path.to_string_lossy().to_string(),
                    content,
                });
            }
        }
    }

    Ok(files)
}

pub fn read_file_with_encoding(path: &Path, encoding: &str) -> Option<String> {
    let content = fs::read(path).ok()?;
    let encoding = Encoding::for_label(encoding.as_bytes()).unwrap_or(UTF_8);

    let (content, has_error) = encoding.decode_with_bom_removal(&content);
    if has_error {
        error!("Error decoding file: {:?}", path);
        return None;
    }

    Some(content.to_string())
}
