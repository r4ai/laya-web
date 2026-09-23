import { readFileSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { expect, it } from "vitest";
import ts from "typescript";

it("keeps documentation links resolvable in the repository and package", () => {
  const pkg = JSON.parse(readFileSync("package.json", "utf8"));
  for (const file of [
    "README.md",
    "docs/development.md",
    "docs/validation.md",
  ]) {
    expect(
      pkg.files.some(
        (entry: string) => file === entry || file.startsWith(`${entry}/`),
      ),
    ).toBe(true);
    const content = readFileSync(file, "utf8");
    expect(content).not.toContain("https://r4ai.github.io/hello-jev/");
    for (const [, link] of content.matchAll(/\]\(([^)]+)\)/g)) {
      if (/^(https?:|#)/.test(link)) continue;
      expect(
        existsSync(resolve(dirname(file), link.split("#")[0])),
        `${file}: ${link}`,
      ).toBe(true);
    }
  }
});

it("typechecks the actual README example against the public API", () => {
  const readme = readFileSync("README.md", "utf8");
  const examples = [...readme.matchAll(/```ts\n([\s\S]*?)```/g)].map(
    (match) => match[1],
  );
  expect(examples.length).toBeGreaterThan(0);
  // Separate virtual modules keep examples independent while sharing one compiler program.
  const sources = new Map(
    examples.map((code, index) => [
      resolve(`readme-example-${index}.ts`),
      code + "\nexport {};\n",
    ]),
  );
  const config = ts.readConfigFile("tsconfig.json", ts.sys.readFile);
  const { options } = ts.parseJsonConfigFileContent(
    config.config,
    ts.sys,
    process.cwd(),
  );
  const host = ts.createCompilerHost(options);
  const original = host.getSourceFile.bind(host);
  host.getSourceFile = (path, version, onError, create) => {
    const code = sources.get(path);
    return code === undefined
      ? original(path, version, onError, create)
      : ts.createSourceFile(path, code, version, true);
  };
  const program = ts.createProgram([...sources.keys()], options, host);
  const errors = ts.getPreEmitDiagnostics(program).map((diagnostic) => ({
    file: diagnostic.file?.fileName,
    message: ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n"),
  }));
  expect(errors).toEqual([]);
});
