import { OpenAI } from "langchain/llms/openai";
import { SerpAPI } from "langchain/tools";
import { initializeAgentExecutor } from "langchain/agents";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { BufferMemory } from "langchain/memory";
import { Calculator } from "langchain/tools/calculator";
import { WebBrowser } from "langchain/tools/webbrowser";

export default class BrowserAgentProvider {
    tools = [new SerpAPI()];

    async init() {
      const CUSTOM_QUESTION_GENERATOR_CHAIN_PROMPT = `
      Given the following conversation and a follow up question, return the conversation history excerpt that includes any relevant context to the question if it exists and rephrase the follow up question to be a standalone question.
      Chat History:
      {chat_history}
      Follow Up Input: {question}
      Your answer should follow the following format:
      <Relevant chat history excerpt as context here>
      Standalone question: <Rephrased question here>
      Your answer:`;
        
        // Initialize OpenAI model
        const model = new OpenAI({ temperature: 0 });

  const vectorStore = await HNSWLib.fromTexts(
    [
      "Aja de forma levemente engraçada e Você deve incluir emojis, seu nome é Felipe e você é um habil vendedor.",
      "Empresa Banho de Ouro 18k\nEndereço, Comendador Soares, Nova Iguaçu - RJ\nFuncionamento: Segunda a segunda, das 10h às 21h.\n🚚 Entrega: Todos os dias, das 10h às 21h.",
      "Informe ao cliente que a empresa não possui uma loja física, sendo exclusivamente de vendas online.\nNão atendemos ao público, não é possível retirar na loja.\nNão fazemos gravação e nossas alianças não são de ouro maciço.",
      "Qualidade e Garantia do Produto:\nChamamos de ouro oco, alianças com revestimento esmaltado em ouro 18k Não ficam pretas e nem descasca Tem garantia de 3 meses são de aço e liga de Prata, revestida e esmaltada em ouro 18 ( é um tipo de banho). Por isso não ficam pretas. Não reagem ao ácido úrico que temos nas mãos, através do suor. Recomendação da fábrica Como qualquer semi jóia. Só tem que evitar contato excessivo com produtos corrosivos, cloro puro, gasolina, tinner. Fora isso, o uso é normal.",
      "Qual modelo de aliança você gostaria de adquirir?\\n\\nGrossa = 10mm, medianas = 8mm, finas = 6, 4 e 2 mm\\\n\n" +
      "Alianças boleadas:\n\n" +
      "1️⃣ [ Cód 1: Par de Alianças em Ouro OCO18k boleadas lisa de 10mm\n💰 R$ 160,00\n📷 Foto: https://wa.me/p/6640280576021922/5521983911694]]\n\n" +
      "2️⃣ [ Cód 2: Par de Alianças em Ouro OCO18k boleadas lisa de 8mm\n💰 R$ 140,00\n📷 Foto: https://wa.me/p/6255029001217544/5521983911694]]\n\n" +
      "3️⃣ [ Cód 3: Par de Alianças em Ouro OCO18k boleadas lisa de 6mm\n💰 R$ 140,00\n📷 Foto: https://wa.me/p/6324607797607300/5521983911694]]\n\n" +
      "4️⃣ [ Cód 4: Par de Alianças em Ouro OCO18k boleadas lisa de 4mm\n💰 R$ 120,00\n📷 Foto: https://wa.me/p/6557431930988859/5521983911694]]\n\n" +
      "🔸 Alianças retas:\n\n" +
      "5️⃣ [ Cód 5: Par de Alianças em Ouro OCO18k retas lisa de 10mm\n💰 R$ 160,00\n📷 Foto: https://wa.me/p/6200381753415676/5521983911694]]\n\n" +
      "6️⃣ [ Cód 6: Par de Alianças em Ouro OCO18k retas lisa de 8mm\n💰 R$ 140,00\n📷 Foto: https://wa.me/p/6209862835801338/5521983911694]]\n\n" +
      "7️⃣ [ Cód 7: Par de Alianças em Ouro OCO18k retas lisa de 6mm\n💰 R$ 140,00\n📷 Foto: https://wa.me/p/6384340114990768/5521983911694]]\n\n" +
      "8️⃣ [ Cód 8: Par de Alianças em Ouro OCO18k retas lisa de 4mm\n💰 R$ 120,00\n📷 Foto: https://wa.me/p/6306925109399349/5521983911694]]\n\n" +
      "Alianças boleadas com anel solitário:\n\n" +
      "1️⃣a [ Cód 1a: Par de Alianças em Ouro OCO18k boleadas lisa de 10mm com anel solitário\n💰 R$ 180,00\n📷 Foto: https://wa.me/p/7033592480002878/5521983911694]]\n\n" +
      "2️⃣a [ Cód 2a: Par de Alianças em Ouro OCO18k boleadas lisa de 8mm com anel solitário\n💰 R$ 160,00\n📷 Foto: https://wa.me/p/6538430342904577/5521983911694]]\n\n" +
      "3️⃣a [ Cód 3a: Par de Alianças em Ouro OCO18k boleadas lisa de 6mm com anel solitário\n💰 R$ 160,00\n📷 Foto: https://wa.me/p/6518642808202070/5521983911694]]\n\n" +
      "4️⃣a [ Cód 4a: Par de Alianças em Ouro OCO18k boleadas lisa de 4mm com anel solitário\n💰 R$ 145,00\n📷 Foto: https://wa.me/p/6482083848520889/5521983911694]]\n\n" +
      "5️⃣a [ Cód 5a: Par de Alianças em Ouro OCO18k retas lisa de 10mm com anel solitário\n💰 R$ 180,00\n📷 Foto: https://wa.me/p/23980852594847420/5521983911694]]\n\n" +
      "6️⃣a [ Cód 6a: Par de Alianças em Ouro OCO18k retas lisa de 8mm com anel solitário\n💰 R$ 160,00\n📷 Foto: https://wa.me/p/6400387426703907/5521983911694]]\n\n" +
      "7️⃣a [ Cód 7a: Par de Alianças em Ouro OCO18k retas lisa de 6mm com anel solitário\n💰 R$ 160,00\n📷 Foto: https://wa.me/p/6482083848520889/5521983911694]]\n\n" +
      "8️⃣a [ Cód 8a: Par de Alianças em Ouro OCO18k retas lisa de 4mm com anel solitário\n💰 R$ 145,00\n📷 Foto: https://wa.me/p/6306925109399349/5521983911694]]\n\n" +
      "\\n\\n\"Se o cliente desejar comprar um unidade o valor do Par de Alianças em Ouro OCO18k deve ser de %50 em relação ao valor do par\".\"\n\n" +
      "Após a escolha, ofereça produtos adicionais:\n\n" +
      "Ofereça a caixinha:\n\n" +
      "\"Caixinha porta alianças em veludo por R$ 15,00\n📷 Foto: https://wa.me/p/6310296232373143/5521983911694]\"\n\n" +
      "Se necessário:\"\n\n" +
      "\"Anel solitário por R$ 30,00\n📷 Foto: https://wa.me/p/6463634063693776/5521983911694]\"\n\n" +
      "Pergunte claramente se o cliente deseja comprar o produto\"\n\n" +
      "Pedido de Troca: vai cobrado o valor de R$ 20,00, minimo para troca em estações de trem da rede SuperVia, outros lugares: consultar a taxa de entrega.\"\n\n" +
      "Taxas de Entrega:\nA entrega é super importante, afinal, queremos que suas alianças cheguem perfeitinhas até você! Temos algumas opções de entrega com suas respectivas taxas:\n- Estação de trem SuperVia: R$ 10,00\n- Estação de metrô: R$ 15,00\n- Entrega residencial: O valor varia de acordo com a distância da estação SuperVia mais próxima:\n  - Até 5km: R$ 20,00\n  - Até 10km: R$ 25,00\n  - Até 15km: R$ 30,00\n- Para bairros de Niterói e região: A taxa de entrega é de R$ 45,00.",
      "Falando em pagar, temos várias opções! Você pode pagar em dinheiro, via Pix ou com cartões de débito e crédito. Só lembrando que se optar por pagar com cartão de crédito na hora da entrega, vai rolar uma taxa extra de R$ 10,00 para cobrir os custos do processamento do pagamento.",
      "Aqui vem uma informação importante! Nossas alianças são exclusivamente vendidas online, ou seja, não temos loja física para retirada. E olha só, elas não são de ouro maciço, tá? de acordo com a opção de entrega podemos enviar todos os tamanho para você experimentar na hora.",
      "Qualidade e Garantia: Nossas alianças são conhecidas como ouro oco, feitas com um revestimento esmaltado em ouro 18k. E o melhor é que elas têm garantia de 3 meses. Vale destacar que elas são feitas de aço e liga de prata, revestidas e esmaltadas em ouro 18k, o que evita que elas escureçam. Ah, e não reagem ao ácido úrico presente no suor das mãos. Só é bom evitar o contato com produtos corrosivos.",
      "Finalização da Venda: Vamos acertar tudo antes de fecharmos a venda, tá? Eu vou reiterar todas as informações do pedido, confirmar com você e gerar um resumo detalhado para garantir que tudo esteja correto. Se houver qualquer alteração, é só me avisar, e a gente ajusta!",
      "Exemplo: Você deve enviar a mensagem e ao gerar o resumo do pedido de acordo com a formatação visual de cada produto.\n\n🔸 [ Cód 3: Par de Alianças em Ouro OCO18k boleadas lisa de 6mm\n\n💰 R$ 140,00\n📷 Foto: https://wa.me/p/6324607797607300/5521983911694].\n\n😀 Essas alianças são elegantes e sofisticadas, ideais para simbolizar o amor e o compromisso entre duas pessoas.\n\n🛒 Posso incluir esse modelo à sua compra?",
  ],
    [{ id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }, { id: 5 }, { id: 6 }, { id: 7 }, { id: 8 }, { id: 9 }, { id: 10 },
      { id: 11 }, { id: 12 }, { id: 13 }, { id: 14 }, { id: 15 }, { id: 16 }, { id: 17 }, { id: 18 }, { id: 19 }, { id: 20 },
      { id: 21 }, { id: 22 }, { id: 23 }, { id: 24 }, { id: 25 }, { id: 26 }, { id: 27 }, { id: 28 }, { id: 29 }, { id: 30 },
      { id: 31 }, { id: 32 }, { id: 33 }, { id: 34 }, { id: 35 }, { id: 36 }, { id: 37 }
  ],
    new OpenAIEmbeddings()
  );
  
    // Create the chain
    const chain = ConversationalRetrievalQAChain.fromLLM(
      new OpenAI({ temperature: 0 }), // Initialize OpenAI model here
      vectorStore.asRetriever(),
      {
          memory: new BufferMemory({
              memoryKey: "chat_history",
              returnMessages: true,
          }),
          questionGeneratorChainOptions: {
              template: CUSTOM_QUESTION_GENERATOR_CHAIN_PROMPT,
          },
      }
  );
  
  const res = await chain.call({
    question: "Quero comprar o modelo Cód 3",
});

console.log(res);

const res2 = await chain.call({
    question: "Quais são as opções recomendadas com anel solitário?"
});

console.log(res2);
}

async fetch(query) {
const executor = await initializeAgentExecutor(this.tools, new OpenAI({ temperature: 0 }), "zero-shot-react-description", true);
const result = await executor.call({ input: query });

return result.output;
}
}
