/**
 * extract-parameters-handheld.mjs
 *
 * Parse một file XML config thiết bị cầm tay (Handheld như Gryphon) để trích xuất Parameter Catalog đầy đủ,
 * bao gồm:
 * - Section (Beeper, Indicators, ...)
 * - Parameter (Good Read Beep Enable, Volume, ...)
 *
 * Hỗ trợ trích xuất:
 * - interfaceDefaults từ các thẻ <value context="..." read="...">
 * - options được map theo thứ tự index của danh sách <value> với CodeTable tương ứng.
 *
 * Usage:
 *   node extract-parameters-handheld.mjs --xml=<path-to-xml> [--section=ReadingParameters] [--out=result.json]
 *
 * Ví dụ:
 *   node extract-parameters-handheld.mjs --xml="xml/config_Gryphon-GM4500_610099261.xml" --out="output/config_Gryphon-GM4500_610099261.json"
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = __dirname;

// ─── CLI args ──────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
const xmlArg     = args.find(a => a.startsWith('--xml='));
const sectionArg = args.find(a => a.startsWith('--section='));
const outArg     = args.find(a => a.startsWith('--out='));

if (!xmlArg) {
  console.error('ERROR: --xml=<path> is required');
  process.exit(1);
}

const XML_PATH   = path.resolve(ROOT, xmlArg.split('=').slice(1).join('='));
const SECTION_FILTER = sectionArg ? sectionArg.split('=').slice(1).join('=').toLowerCase() : null;
const OUTPUT     = outArg ? outArg.split('=').slice(1).join('=') : null;

if (!fs.existsSync(XML_PATH)) {
  console.error(`ERROR: File not found: ${XML_PATH}`);
  process.exit(1);
}

const xmlContent = fs.readFileSync(XML_PATH, 'utf-8');
const deviceName = XML_PATH.match(/config_(.+)\.xml/)?.[1] ?? path.basename(XML_PATH, '.xml');

// ─── Helper: parse XML attributes string ──────────────────────────────────
function parseAttrs(attrsStr) {
  const attrs = {};
  const re = /(\w[\w-]*)="([^"]*)"/g;
  let m;
  while ((m = re.exec(attrsStr)) !== null) attrs[m[1]] = m[2];
  return attrs;
}

// ─── 0. Parse interface class names (dùng làm key cho interfaceDefaults) ─
const interfaceNames = [];
const interfaceClassRE = /<interfaceclass\s+name="([^"]+)"/g;
let icm;
while ((icm = interfaceClassRE.exec(xmlContent)) !== null) {
  if (!interfaceNames.includes(icm[1])) interfaceNames.push(icm[1]);
}

// Fallback for Handheld: Parse interface names from Interface.Selection values
if (interfaceNames.length === 0) {
  const interfaceSelectionRE = /<parameter\s+[^>]*name="Interface\.Selection"[^>]*>([\s\S]*?)<\/parameter>/;
  const isMatch = xmlContent.match(interfaceSelectionRE);
  if (isMatch) {
    const valueRE = /<value\s+([^>]+)>/g;
    let valMatch;
    while ((valMatch = valueRE.exec(isMatch[1])) !== null) {
      const attrsStr = valMatch[1].endsWith('/') ? valMatch[1].slice(0, -1) : valMatch[1];
      const attrs = parseAttrs(attrsStr);
      if (attrs.context && !interfaceNames.includes(attrs.context)) {
        interfaceNames.push(attrs.context);
      }
    }
  }
}

function extractInterfaceDefaults(innerContent) {
  const interfaceDefaults = {};

  // For Gryphon/handheld format: find all <value ...> elements and check if they have context attribute
  const valueRE = /<value\s+([^>]+)>/g;
  let valMatch;
  while ((valMatch = valueRE.exec(innerContent)) !== null) {
    const attrsStr = valMatch[1].endsWith('/') ? valMatch[1].slice(0, -1) : valMatch[1];
    const attrs = parseAttrs(attrsStr);
    if (attrs.context) {
      interfaceDefaults[attrs.context] = attrs.read || '';
    }
  }

  // Fallback to Magellan format if the script is ever run on Magellan XML
  if (Object.keys(interfaceDefaults).length === 0) {
    for (const name of interfaceNames) {
      const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const tagRE = new RegExp(`<${escaped}>([\\s\\S]*?)<\\/${escaped}>`);
      const match = innerContent.match(tagRE);
      if (match) {
        interfaceDefaults[name] = match[1].trim();
      }
    }

    // Fallback format cũ (nếu có)
    const interfaceAttrRE = /<interface\s+name="([^"]+)"\s+value="([^"]*)"\s*\/>/g;
    let interfaceMatch;
    while ((interfaceMatch = interfaceAttrRE.exec(innerContent)) !== null) {
      if (interfaceDefaults[interfaceMatch[1]] === undefined) {
        interfaceDefaults[interfaceMatch[1]] = interfaceMatch[2];
      }
    }
  }

  return interfaceDefaults;
}

// ─── 1. Parse Tables (CodeTable → options) ────────────────────────────────
const tableMap = {}; // tableName → [{name, value}]
const tableRE = /<table\s+name="([^"]+)">([\s\S]*?)<\/table>/g;
const elementRE = /<element(?:\s+name="([^"]*)")?>([\s\S]*?)<\/element>/g;

let tm;
while ((tm = tableRE.exec(xmlContent)) !== null) {
  const tableName = tm[1];
  const tableBody = tm[2];
  const elements = [];
  let em;
  elementRE.lastIndex = 0;
  while ((em = elementRE.exec(tableBody)) !== null) {
    elements.push({ name: em[1] ?? '', value: em[2].trim() });
  }
  elementRE.lastIndex = 0;
  tableMap[tableName] = elements;
}

// ─── 1b. Parse exeNumericRange (min/max cho tableRef không phải CodeTable) ─
const numericRangeMap = {};
const numericRangeRE = /<exeNumericRange\s+([^/>]+)\/?>/g;
let nrm;
while ((nrm = numericRangeRE.exec(xmlContent)) !== null) {
  const attrs = parseAttrs(nrm[1]);
  if (attrs['name']) {
    numericRangeMap[attrs['name']] = {
      min: attrs['min'] ?? '',
      max: attrs['max'] ?? '',
    };
  }
}

function isCodeTable(tableRef) {
  return !!tableRef && tableRef.startsWith('CodeTable');
}

function getOptions(tableRef, paramOptions) {
  const tableElements = tableRef ? (tableMap[tableRef] ?? []) : [];
  const options = {};
  
  if (paramOptions && paramOptions.length > 0) {
    // Map parameter-specific labels to table values by index
    for (let i = 0; i < paramOptions.length; i++) {
      if (i < tableElements.length) {
        options[paramOptions[i]] = tableElements[i].value;
      }
    }
  } else {
    // Use table's element names directly
    for (const el of tableElements) {
      const name = el.name || el.value;
      options[name] = el.value;
    }
  }
  return options;
}

function buildTableConstraints(attrs, paramOptions) {
  const tableRef = attrs['tableRef'] ?? '';
  if (!tableRef) return {};

  if (isCodeTable(tableRef)) {
    const opts = getOptions(tableRef, paramOptions);
    if (Object.keys(opts).length > 0) {
      return { options: opts };
    }
    return {};
  }

  const rangeDef = numericRangeMap[tableRef] ?? {};
  const min = attrs['min'] ?? rangeDef.min;
  const max = attrs['max'] ?? rangeDef.max;
  const constraints = {};

  if (attrs['min'] !== undefined) constraints.min = attrs['min'];
  if (attrs['max'] !== undefined) constraints.max = attrs['max'];
  if (attrs['incrementBy'] !== undefined) constraints.incrementBy = attrs['incrementBy'];

  if (min !== undefined && max !== undefined) {
    constraints.range = { min, max };
  }

  return constraints;
}

// ─── 3. Parse Parameters từ <parameters> section ──────────────────────────
const paramMap = {};
const paramDefRE = /<(parameter|singleParameter)\s+([^>]+)>([\s\S]*?)<\/\1>/g;
const contextRE = /<context>([\s\S]*?)<\/context>/;

let m;
while ((m = paramDefRE.exec(xmlContent)) !== null) {
  const attrs = parseAttrs(m[2]);
  const innerContent = m[3];
  const contextMatch = innerContent.match(contextRE);
  const label = contextMatch ? contextMatch[1].trim() : (attrs['name'] ?? '');

  if (attrs['name']) {
    // Extract param-specific value options (labels) without attributes
    const paramOptions = [];
    const valTextRE = /<value\s*>([^<]+)<\/value>/g;
    let vtMatch;
    while ((vtMatch = valTextRE.exec(innerContent)) !== null) {
      paramOptions.push(vtMatch[1].trim());
    }

    paramMap[attrs['name']] = {
      value:        attrs['value'] ?? '',
      type:         attrs['type'] ?? '',
      protection:   attrs['protection'] ?? '',
      code:         attrs['code'] ?? '',
      sizeLen:      attrs['sizeLen'] ?? '',
      tableRef:     attrs['tableRef'] ?? '',
      context:      label,
      interfaceDefaults: extractInterfaceDefaults(innerContent),
      ...buildTableConstraints(attrs, paramOptions),
    };
  }
}

// ─── 4. Parse Pages ────────────────────────────────────────────────────────
function parsePages(xmlStr) {
  const tokens = [];
  const pageOpenRE = /<page\s+([^>]+)>/g;
  const pageCloseRE = /<\/page>/g;
  let mo;

  pageOpenRE.lastIndex = 0;
  while ((mo = pageOpenRE.exec(xmlStr)) !== null) {
    tokens.push({ type: 'open', pos: mo.index, endPos: mo.index + mo[0].length, attrs: parseAttrs(mo[1]) });
  }
  pageCloseRE.lastIndex = 0;
  while ((mo = pageCloseRE.exec(xmlStr)) !== null) {
    tokens.push({ type: 'close', pos: mo.index, endPos: mo.index + mo[0].length });
  }
  tokens.sort((a, b) => a.pos - b.pos);

  const stack = [];
  const rootPages = [];

  for (const token of tokens) {
    if (token.type === 'open') {
      const node = {
        name:       token.attrs['name'] ?? '',
        title:      token.attrs['title'] ?? '',
        tocId:      token.attrs['tocId'] ?? '',
        protection: token.attrs['protection'] ?? '',
        startPos:   token.pos,
        endPos:     token.endPos,
        children:   [],
        fields:     [],
        labels:     [],
      };
      if (stack.length > 0) stack[stack.length - 1].children.push(node);
      else rootPages.push(node);
      stack.push(node);
    } else if (token.type === 'close') {
      if (stack.length > 0) {
        const node = stack.pop();
        node.closePos = token.endPos;
        const pageContent = xmlStr.slice(node.endPos, node.closePos);

        const fieldRE = /<field\s+name="([^"]+)"/g;
        const labelRE = /<label\s+name="([^"]+)"\s+text="([^"]+)"/g;
        let f, l;

        fieldRE.lastIndex = 0;
        while ((f = fieldRE.exec(pageContent)) !== null) {
          const before = pageContent.slice(0, f.index);
          if ((before.match(/<page\s/g) || []).length === (before.match(/<\/page>/g) || []).length) {
            node.fields.push(f[1]);
          }
        }
        labelRE.lastIndex = 0;
        while ((l = labelRE.exec(pageContent)) !== null) {
          const before = pageContent.slice(0, l.index);
          if ((before.match(/<page\s/g) || []).length === (before.match(/<\/page>/g) || []).length) {
            node.labels.push({ name: l[1], text: l[2] });
          }
        }
      }
    }
  }
  return rootPages;
}

const pages = parsePages(xmlContent);

// ─── 5. Build catalog (key = page.name từ XML, vd: ReadingParameters.Beeper.pnl) ─
function buildCatalog(pages) {
  const catalog = {};
  for (const page of pages) {
    const pageObject = {
      title: page.title,
      tocId: page.tocId,
      protection: page.protection,
    };

    for (const fieldName of page.fields) {
      const param = paramMap[fieldName];
      if (param) {
        pageObject[fieldName] = {
          value: param.value,
          type: param.type,
          protection: param.protection,
          code: param.code,
          sizeLen: param.sizeLen,
          tableRef: param.tableRef,
          ...(param.min !== undefined ? { min: param.min } : {}),
          ...(param.max !== undefined ? { max: param.max } : {}),
          ...(param.incrementBy !== undefined ? { incrementBy: param.incrementBy } : {}),
          context: param.context,
          interfaceDefaults: param.interfaceDefaults,
          ...(param.options !== undefined ? { options: param.options } : {}),
          ...(param.range !== undefined ? { range: param.range } : {}),
        };
      }
    }
    catalog[page.name] = pageObject;
    Object.assign(catalog, buildCatalog(page.children));
  }
  return catalog;
}

const finalCatalog = {};
for (const rootPage of pages) {
    Object.assign(finalCatalog, buildCatalog([rootPage]));
}

const json = JSON.stringify(finalCatalog, null, 2);

if (OUTPUT) {
  fs.writeFileSync(path.resolve(ROOT, OUTPUT), json, 'utf-8');
  console.log(`Wrote catalog to ${OUTPUT}`);
} else {
  console.log(json);
}
