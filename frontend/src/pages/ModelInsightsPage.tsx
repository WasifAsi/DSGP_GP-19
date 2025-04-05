import { useEffect, useState } from "react";
import { FiSend } from "react-icons/fi";
import { BsRobot } from "react-icons/bs";
import { motion } from "framer-motion";

type Message = {
  text?: string;
  image?: string;
  sender: "bot" | "user";
  time?: string;
};

const TypingIndicator = () => (
  <div className="flex items-center space-x-1">
    {[0, 1, 2].map((i) => (
      <motion.span
        key={i}
        className="w-2 h-2 bg-muted-foreground rounded-full"
        animate={{ opacity: [0.3, 1, 0.3] }}
        transition={{
          duration: 1,
          repeat: Infinity,
          ease: "easeInOut",
          delay: i * 0.2,
        }}
      />
    ))}
  </div>
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
    {
      text: "Hey there 👋 How can I help you today?",
      sender: "bot",
      time: " ",
    },
  ]);
  const [input, setInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

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

      setMessages((prev) => [...prev.slice(0, -1), ...botResponses]);
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
    } finally {
      setLoading(false);
    }
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
            <ChatMessage
              key={index}
              text={msg.text}
              image={msg.image}
              sender={msg.sender}
              time={msg.time}
            />
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
          <div className="mt-3 flex gap-2 text-xs flex-wrap">
            {["What is U-Net?", "Predict shoreline", "📊 FAQs"].map((text, idx) => (
              <button
                key={idx}
                onClick={() => {
                  if (!loading) {
                    setInput(text);
                    handleSend();
                  }
                }}
                className={`${
                  loading ? "opacity-50 cursor-not-allowed" : "hover:bg-accent/80"
                } bg-accent text-accent-foreground rounded-full px-3 py-1 transition`}
                disabled={loading}
              >
                {text}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

};

export default ModelInsightsPage;
