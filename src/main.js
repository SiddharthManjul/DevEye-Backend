import * as dotenv from "dotenv";
dotenv.config();

import { TaskType } from "@google/generative-ai";

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";

import { compile } from "html-to-text";
import { RecursiveUrlLoader } from "@langchain/community/document_loaders/web/recursive_url";

import promptSync from "prompt-sync";
const promptFromTerminal = promptSync({ sigint: true });

// Model Initialization
const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: "gemini-pro",
  temperature: 0,
  // maxOutputTokens: 2048,
});

// String Parser from Langchain
const parser = new StringOutputParser();

// Regex Parser
// const codeBlockRegex = /```(.*?)```/s;
// const regexCodeBlockParser = new RegexParser(codeBlockRegex, 1);

// Custom Regex Parser
const parseCodeBlockUsingRegex = (text) => {
  const codeBlockRegex = /```(.*?)```/gs;
  const matches = [];
  let match;
  while ((match = codeBlockRegex.exec(text)) !== null) {
    matches.push(match[1].replace(/\\n/g, "\n").replace(/\\"/g, '"').trim());
  }
  return matches.length > 0 ? matches : null;
};

// Function to format content
const formatContent = (text) => {
  const codeBlockRegex = /```(.*?)```/gs;
  return text.replace(
    codeBlockRegex,
    (_, code) =>
      `\nCode Snippet:\n${code
        .replace(/\\n/g, "\n")
        .replace(/\\"/g, '"')
        .trim()}\n`
  );
};

// Prompt Template
const prompt = ChatPromptTemplate.fromMessages([
  `
    Answer the User Question. Use the context provided only. If nothing found tell them politely that you don't have this information available
    Context: {context}
    Question: {input}    
`,
]);

// Document Loader
const loader = new CheerioWebBaseLoader(
  "https://doc.rust-lang.org/book/ch01-02-hello-world.html"
);

const url = "https://nextjs.org/";

const compiledConvert = compile({ wordwrap: 3000 }); // returns (text: string) => string;

const recursiveLoader = new RecursiveUrlLoader(url, {
  extractor: compiledConvert,
  maxDepth: 100,
  excludeDirs: ["https://doc.rust-lang.org/book/title-page.html", "https://doc.rust-lang.org/book/foreword.html"],
});

const docs = await recursiveLoader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 2000,
  chunkOverlap: 200,
});

const splitDocs = await splitter.splitDocuments(docs);

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: "embedding-001",
  taskType: TaskType.RETRIEVEL_DOCUMENT,
});

const vecotrStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);

// Retrieve Data
const retriever = vecotrStore.asRetriever({
  k: 0,
});

// Output Formatting
const isCode = (text) => {
  const codePatterns = [
    /^(\$|>)/,
    /^#!/,
    /;$/,
    /\bfunction\b/,
    /\bconst\b/,
    /\blet\b/,
    /\bvar\b/,
  ];

  return codePatterns.some((pattern) => pattern.test(text));
};

const input = promptFromTerminal("How can I help you? ");

// Runner Function
const run = async () => {
  // const chain = prompt.pipe(model);
  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt: prompt,
  });

  const parsedChain = chain.pipe(parser);

  const retrievalChain = await createRetrievalChain({
    combineDocsChain: parsedChain,
    retriever: retriever,
  });

  const retrievedResponse = await retrievalChain.invoke({
    input: input,
  });

  const response = JSON.stringify(retrievedResponse, null, 2);
  // console.log(response);

  // const message = retrievedResponse;

  const formattedResponse = formatContent(response);
  // console.log(`\n${formattedResponse}\n`);

  // Extract the answer
  const responseObject = JSON.parse(response);
  const answer = responseObject.answer;

  // Print the answer in green color
  const green = '\x1b[32m%s\x1b[0m';
  console.log(green, `\n${answer}\n`);
};

run();
