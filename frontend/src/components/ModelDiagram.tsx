import { motion } from 'framer-motion';
import { CameraIcon, Database, BarChart3, BrainCircuit, Image } from 'lucide-react';

const ModelDiagram = () => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.8 }}
      className="relative w-full h-[400px] md:h-[500px]"
    >
      {/* Central brain node */}
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.8 }}
        className="absolute top-[40%] left-[42.5%] transform -translate-x-1/2 -translate-y-1/2 z-10"
        style={{
          transform: 'translate(-50%, -50%)',
          position: 'absolute',
        }}
      >
        <motion.div
          animate={{ 
            scale: [0.98, 1.02, 0.98] 
          }}
          transition={{ 
            duration: 4, 
            repeat: Infinity, 
            repeatType: "loop" 
          }}
          className="w-24 h-24 rounded-full bg-gradient-to-br from-shoreline-blue to-blue-400 flex items-center justify-center shadow-lg shadow-shoreline-blue/20"
        >
          <BrainCircuit size={40} className="text-white" />
        </motion.div>
        <div className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 text-center">
          <p className="font-medium text-shoreline-dark dark:text-white">AI Core</p>
        </div>
      </motion.div>

      {/* Connection lines */}
      {[30, 90, 150, 210, 270, 330].map((angle, index) => (
        <motion.div
          key={angle}
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.6 }}
          transition={{ duration: 0.5, delay: 0.8 + index * 0.1 }}
          className="absolute top-1/2 left-1/2 w-[150px] h-[2px] bg-gradient-to-r from-transparent via-shoreline-blue to-transparent"
          style={{ 
            transform: `translate(-50%, -50%) rotate(${angle}deg)`,
            transformOrigin: 'center'
          }}
        />
      ))}

      {/* Surrounding nodes */}
      {itemVariants.map((item, index) => (
        <motion.div
          key={index}
          initial="hidden"
          animate="visible"
          custom={index}
          className="absolute"
          style={{ 
            top: `${item.top}%`, 
            left: `${item.left}%`,
            transform: 'translate(-50%, -50%)'
          }}
        >
          <motion.div
            initial={{ scale: 0, opacity: 0 }}
            animate={{ 
              scale: 1, 
              opacity: 1 
            }}
            transition={{
              delay: 0.3 + index * 0.15,
              duration: 0.6,
              type: "spring",
              stiffness: 200,
              damping: 15
            }}
            className={`w-16 h-16 rounded-full flex items-center justify-center shadow-md ${item.bgClass}`}
          >
            {item.icon}
          </motion.div>
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 text-center w-28">
            <p className="text-sm font-medium text-shoreline-dark dark:text-white">{item.label}</p>
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
};

// Item configurations
const itemVariants = [
  {
    top: 25,
    left: 25,
    label: 'Image Input',
    bgClass: 'bg-blue-100 dark:bg-blue-900',
    icon: <CameraIcon size={24} className="text-blue-600 dark:text-blue-400" />
  },
  {
    top: 25,
    left: 75,
    label: 'Image Processing',
    bgClass: 'bg-purple-100 dark:bg-purple-900',
    icon: <Image size={24} className="text-purple-600 dark:text-purple-400" />
  },
  {
    top: 50,
    left: 12,
    label: 'Data Storage',
    bgClass: 'bg-green-100 dark:bg-green-900',
    icon: <Database size={24} className="text-green-600 dark:text-green-400" />
  },
  {
    top: 50,
    left: 88,
    label: 'Analytics',
    bgClass: 'bg-orange-100 dark:bg-orange-900',
    icon: <BarChart3 size={24} className="text-orange-600 dark:text-orange-400" />
  },
  {
    top: 75,
    left: 30,
    label: 'CNN Model',
    bgClass: 'bg-shoreline-light-blue dark:bg-blue-900',
    icon: <BrainCircuit size={24} className="text-shoreline-blue dark:text-blue-400" />
  },
  {
    top: 75,
    left: 70,
    label: 'LSTM Model',
    bgClass: 'bg-indigo-100 dark:bg-indigo-900',
    icon: <BrainCircuit size={24} className="text-indigo-600 dark:text-indigo-400" />
  }
];

export default ModelDiagram;
