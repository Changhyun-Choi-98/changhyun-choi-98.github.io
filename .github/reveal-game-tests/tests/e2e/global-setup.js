const path = require("node:path");
const { pathToFileURL } = require("node:url");

module.exports = async () => {
  const modulePath = path.resolve(__dirname, "../../scripts/generate-fixtures.mjs");
  const { generateFixtures } = await import(pathToFileURL(modulePath));
  await generateFixtures();
};
