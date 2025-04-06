import { useState } from "react";
import { FiSend } from "react-icons/fi";
import { BsRobot } from "react-icons/bs";
import { motion, AnimatePresence } from "framer-motion";


type Message = {
  text?: string;
  image?: string;
  sender: "bot" | "user";
  time?: string;
};

const TypingIndicator = () => (
  <motion.div
    className="flex items-center space-x-1 px-3 py-2 bg-muted rounded-xl shadow-inner"
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    transition={{ duration: 0.3 }}
  >

    {[0, 1, 2].map((i) => (
      <motion.span
        key={i}
        className="w-2 h-2 bg-muted-foreground rounded-full"

        animate={{
          y: [0, -4, 0],
          opacity: [0.5, 1, 0.5],
        }}
        transition={{
          duration: 0.6,
          repeat: Infinity,
          ease: "easeInOut",
          delay: i * 0.2,
        }}
      />
    ))}
  </motion.div>
);

const ChatMessage = ({ text, image, sender, time }: Message) => {
  const isBot = sender === "bot";

  const avatar = isBot
    ? "https://cdn-icons-png.flaticon.com/512/4712/4712106.png"
    : "https://cdn-icons-png.flaticon.com/512/1077/1077012.png";

  const glowStyle = isBot
    ? "drop-shadow-[0_0_6px_rgba(0,255,150,0.6)] dark:drop-shadow-[0_0_6px_rgba(0,200,255,0.8)]"
    : "drop-shadow-[0_0_6px_rgba(255,200,0,0.5)] dark:drop-shadow-[0_0_6px_rgba(255,255,255,0.7)]";

  const bubbleStyle = isBot
    ? "bg-gradient-to-br from-accent to-primary text-accent-foreground rounded-xl rounded-bl-none shadow-md"
    : "bg-gradient-to-br from-[#cceeff] to-[#61dafb] text-gray-800 rounded-xl rounded-br-none shadow-md dark:from-[#0ea5e9] dark:to-[#0369a1] dark:text-white";

  return (
    <motion.div
      initial={{ opacity: 0, y: 30, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className={`flex items-start ${isBot ? "justify-start" : "justify-end"} mb-4`}
    >
      {isBot && (
        <motion.img
          src={avatar}
          alt="avatar"
          className={`w-8 h-8 rounded-full ${glowStyle} mr-2 shadow`}
        />
      )}
      <div className={`max-w-[75%] px-4 py-3 text-base ${bubbleStyle}`}>
        {text === "Typing..." ? (
          <TypingIndicator />
        ) : (
          <>
            {text && <p>{text}</p>}
            {image && (
              <img
                src={image}
                alt="Bot response"
                className="mt-2 rounded-lg max-w-full max-h-[250px] object-contain"
              />
            )}
          </>
        )}
        {text !== "Typing..." && (
          <div className="text-[10px] text-muted-foreground mt-1 text-right">{time}</div>
        )}
      </div>
      {!isBot && (
        <motion.img
          src={avatar}
          alt="avatar"
          className={`w-8 h-8 rounded-full ${glowStyle} ml-2 shadow`}
        />
      )}
    </motion.div>
  );
};

const ModelInsightsPage: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([

    { text: "Hey there 👋 How can I help you today?", sender: "bot", time: " " },
  ]);
  const [input, setInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [openSection, setOpenSection] = useState<string | null>(null);


  const getCurrentTime = () => {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg: Message = {
      text: input,
      sender: "user",
      time: getCurrentTime(),
    };

    const thinkingMsg: Message = {
      text: "Typing...",
      sender: "bot",
      time: getCurrentTime(),
    };

    setMessages((prev) => [...prev, userMsg, thinkingMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("http://localhost:5005/webhooks/rest/webhook", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sender: "user", message: userMsg.text }),
      });
      const data = await res.json();

      const botResponses: Message[] = data.map((msg: any) => ({
        text: msg.text,
        image: msg.image,
        sender: "bot",
        time: getCurrentTime(),
      }));


      setTimeout(() => {
        setMessages((prev) => [...prev.slice(0, -1), ...botResponses]);
        setLoading(false);
      }, 1000 + botResponses.length * 300);

    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev.slice(0, -1),
        {
          text: "Oops! Couldn't reach the server.",
          sender: "bot",
          time: getCurrentTime(),
        },
      ]);
      setLoading(false);
    }
  };

  const quickReplyGroups: Record<string, string[]> = {
    "🌊 Shoreline": [
      "What is shoreline analysis?",
      "Why is shoreline monitoring important?",
      "Show me shoreline change",
      "Tools for coastal mapping",
      "Sea level rise impact?",
      "Visualize shoreline shift",
      "What causes shoreline change?",
    ],
    "📐 Metrics": [
      "What is NSM?",
      "What is EPR?",
      "NSM vs EPR?",
      "What does negative EPR mean?",
      "How accurate is shoreline detection?",
      "How are transects used?",
      "How many transects to draw?",
    ],
    "🤖 Models": ["What is U-Net?", "Explain SegNet", "DeepLabV3 use cases?", "What is FCN-8?"],
    "💬 General": ["What can you do?", "Explain your purpose", "Who made you?", "Help with shoreline studies?"],
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] mt-16 flex items-center justify-center bg-background px-4 transition-colors duration-300">
      <div className="w-full max-w-2xl bg-card text-card-foreground rounded-3xl shadow-2xl flex flex-col overflow-hidden h-[92vh] border border-border dark:border-white/10">
        {/* Header */}
        <div className="bg-gradient-to-r from-primary to-accent text-primary-foreground px-5 py-4 flex items-center gap-3 rounded-t-3xl">
          <BsRobot className="text-2xl" />
          <div>
            <p className="font-bold text-lg">CoastiBot</p>
            <div className="flex items-center gap-1">
              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <p className="text-xs text-green-400">Online</p>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 p-6 overflow-y-auto scrollbar-thin bg-card transition-colors">
          {messages.map((msg, index) => (
            <ChatMessage key={index} {...msg} />
          ))}
        </div>

        {/* Footer */}
        <div className="p-4 border-t bg-background border-border dark:border-white/10">
          <div className="flex items-center bg-card border border-border dark:border-white/10 rounded-full px-4 py-2 shadow-md">
            <input
              type="text"
              className="flex-1 text-base outline-none bg-transparent text-foreground"
              placeholder="Type your message here..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              disabled={loading}
            />
            <button
              className="ml-3 text-primary hover:brightness-110"
              onClick={handleSend}
              disabled={loading}
            >
              <FiSend className="w-5 h-5" />
            </button>
          </div>

          {/* Quick Replies */}
          <div className="mt-4 space-y-2">
            {Object.entries(quickReplyGroups).map(([label, replies]) => (
              <div key={label}>
                <button
                  className="w-full text-left text-sm font-medium bg-muted px-4 py-2 rounded-xl hover:bg-muted/80 transition"
                  onClick={() => setOpenSection(openSection === label ? null : label)}
                >
                  {label} {openSection === label ? "▲" : "▼"}
                </button>

                <AnimatePresence>
                  {openSection === label && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="flex flex-wrap gap-2 mt-2 px-1"
                    >
                      {replies.map((text, idx) => (
                        <motion.button
                          key={idx}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: idx * 0.05 }}
                          onClick={() => {
                            if (!loading) {
                              setInput(text);
                              handleSend();
                            }
                          }}
                          className={`${
                            loading ? "opacity-50 cursor-not-allowed" : "hover:bg-accent/80"
                          } bg-accent text-accent-foreground rounded-full px-3 py-1 text-xs transition`}
                          disabled={loading}
                        >
                          {text}
                        </motion.button>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelInsightsPage;
