import { useState } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'sonner';

const ContactForm = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: '',
  });
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    toast.success('Message sent successfully!', {
      description: 'We will get back to you as soon as possible.',
    });
    
    setFormData({
      name: '',
      email: '',
      subject: '',
      message: '',
    });
    
    setIsSubmitting(false);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <label htmlFor="name" className="block text-sm font-medium text-shoreline-dark dark:text-white mb-1">
          Name
        </label>
        <input
          id="name"
          name="name"
          type="text"
          required
          placeholder="Your name"
          value={formData.name}
          onChange={handleChange}
          className="w-full px-4 py-3 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-shoreline-dark dark:text-white focus:border-shoreline-blue dark:focus:border-shoreline-blue focus:ring-1 focus:ring-shoreline-blue focus:outline-none transition-colors"
        />
      </div>
      
      <div>
        <label htmlFor="email" className="block text-sm font-medium text-shoreline-dark dark:text-white mb-1">
          Email
        </label>
        <input
          id="email"
          name="email"
          type="email"
          required
          placeholder="Your email address"
          value={formData.email}
          onChange={handleChange}
          className="w-full px-4 py-3 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-shoreline-dark dark:text-white focus:border-shoreline-blue dark:focus:border-shoreline-blue focus:ring-1 focus:ring-shoreline-blue focus:outline-none transition-colors"
        />
      </div>
      
      <div>
        <label htmlFor="subject" className="block text-sm font-medium text-shoreline-dark dark:text-white mb-1">
          Subject
        </label>
        <input
          id="subject"
          name="subject"
          type="text"
          required
          placeholder="What is this regarding?"
          value={formData.subject}
          onChange={handleChange}
          className="w-full px-4 py-3 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-shoreline-dark dark:text-white focus:border-shoreline-blue dark:focus:border-shoreline-blue focus:ring-1 focus:ring-shoreline-blue focus:outline-none transition-colors"
        />
      </div>
      
      <div>
        <label htmlFor="message" className="block text-sm font-medium text-shoreline-dark dark:text-white mb-1">
          Message
        </label>
        <textarea
          id="message"
          name="message"
          rows={5}
          required
          placeholder="Your message..."
          value={formData.message}
          onChange={handleChange}
          className="w-full px-4 py-3 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-shoreline-dark dark:text-white focus:border-shoreline-blue dark:focus:border-shoreline-blue focus:ring-1 focus:ring-shoreline-blue focus:outline-none resize-none transition-colors"
        />
      </div>
      
      <motion.button
        type="submit"
        disabled={isSubmitting}
        whileTap={{ scale: 0.98 }}
        className="w-full bg-shoreline-blue text-white font-medium py-3 px-6 rounded-lg shadow-md hover:shadow-lg hover:shadow-shoreline-blue/20 transition-all duration-300 disabled:opacity-70 disabled:cursor-not-allowed"
      >
        {isSubmitting ? 'Sending...' : 'Send Message'}
      </motion.button>
    </form>
  );
};

export default ContactForm;