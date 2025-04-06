import { useState, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Sun, Moon, ChevronRight } from 'lucide-react';

const Navbar = () => {
  const [isDark, setIsDark] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDark]);

  const toggleTheme = () => {
    setIsDark(!isDark);
  };

  const handleNavigation = (path: string) => {
    window.location.href = path; // Force a page refresh
  };

  const navLinks = [
    { name: 'Home', path: '/' },
    { name: 'Upload & Analysis', path: '/upload' },
    { name: 'Model Insights', path: '/model-insights' },
    { name: 'About Us', path: '/about' },
  ];

  return (
    <motion.header 
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.7, ease: [0.23, 1, 0.32, 1] }}
      className={`fixed top-0 left-0 right-0 z-50 py-4 transition-all duration-500 ${
        isScrolled 
          ? 'bg-white/80 dark:bg-shoreline-dark/80 shadow-lg backdrop-blur-xl border-b border-gray-200/20 dark:border-gray-700/20' 
          : 'bg-transparent'
      }`}
    >
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between">
          <Link 
            to="/" 
            className="group flex items-center space-x-2 relative"
          >
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
              className="relative"
            >
              <span className="text-shoreline-blue font-display font-bold text-2xl">
                Shoreline
                <span className="text-shoreline-dark dark:text-white">Analysis</span>
              </span>
              <motion.div
                className="absolute bottom-0 left-0 w-full h-0.5 bg-shoreline-blue/20"
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              />
            </motion.div>
          </Link>

          <nav className="hidden md:flex items-center space-x-10">
            {navLinks.map((link) => (
              <button
                key={link.path}
                onClick={() => handleNavigation(link.path)}
                className="relative group py-2"
              >
                <span className={`text-sm font-medium transition-colors ${
                  location.pathname === link.path
                    ? 'text-shoreline-blue'
                    : 'text-shoreline-dark/80 dark:text-white/80 group-hover:text-shoreline-blue dark:group-hover:text-shoreline-blue'
                }`}>
                  {link.name}
                </span>
                
                {location.pathname === link.path ? (
                  <motion.div
                    layoutId="navbar-indicator"
                    className="absolute -bottom-1 left-0 right-0 h-0.5 bg-shoreline-blue"
                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                  />
                ) : (
                  <div className="absolute -bottom-1 left-0 right-0 h-0.5 bg-shoreline-blue scale-x-0 group-hover:scale-x-100 transition-transform duration-300" />
                )}
              </button>
            ))}
          </nav>

          <div className="flex items-center space-x-6">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={toggleTheme}
              className="p-2.5 rounded-full bg-gray-100 dark:bg-gray-800 text-shoreline-dark dark:text-white 
                       shadow-md hover:shadow-lg transition-all duration-300"
              aria-label="Toggle dark mode"
            >
              <AnimatePresence mode="wait">
                <motion.div
                  key={isDark ? 'dark' : 'light'}
                  initial={{ opacity: 0, rotate: -180 }}
                  animate={{ opacity: 1, rotate: 0 }}
                  exit={{ opacity: 0, rotate: 180 }}
                  transition={{ duration: 0.3 }}
                >
                  {isDark ? <Sun size={18} /> : <Moon size={18} />}
                </motion.div>
              </AnimatePresence>
            </motion.button>

            <Link 
              to="/upload"
              className="upload-btn group flex items-center"
            >
              Upload Image
              <ChevronRight className="ml-2 w-4 h-4 transition-transform duration-300 group-hover:translate-x-1" />
            </Link>
          </div>
        </div>
      </div>
    </motion.header>
  );
};

export default Navbar;