import { Storage } from 'react-jhipster';
import React, { useEffect, useMemo, useState, useRef } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import {
  CssBaseline,
  ThemeProvider,
  Icon,
  Tooltip,
  TextField,
  IconButton,
  Avatar,
  Typography,
  Box,
} from '@mui/material';
import createCache from '@emotion/cache';
import rtlPlugin from 'stylis-plugin-rtl';
import { CacheProvider } from '@emotion/react';
import axios from 'axios';

import Sidenav from 'app/template1/examples/Sidenav';
import Configurator from 'app/template1/examples/Configurator';
import DashboardNavbar from 'app/template1/examples/Navbars/DashboardNavbar';
import MDBox from 'app/template1/components/MDBox';

import createLightTheme from 'app/template1/assets/theme';
import createDarkTheme from 'app/template1/assets/theme-dark';

import { useMaterialUIController, setOpenConfigurator, setMiniSidenav } from 'app/template1/context';
import { setActivePalette, getActivePaletteKey } from 'app/template1/assets/theme/base/paletteRegister';

import brandWhite from 'app/template1/assets/images/logo-ct.png';
import brandDark from 'app/template1/assets/images/logo-ct-dark.png';
import { useAppSelector } from 'app/config/store';
import transformMenuToRoutes from 'app/utils/transformMenuToRoute';
import ReactMarkdown from 'react-markdown';

const FullLayout = () => {
  const [controller, dispatchUI] = useMaterialUIController();
  const {
    miniSidenav,
    sidenavClose,
    direction,
    layout,
    openConfigurator,
    sidenavColor,
    transparentSidenav,
    whiteSidenav,
    darkMode,
    themeColor,
  } = controller;

  const [rtlCache, setRtlCache] = useState(null);
  const [onMouseEnter, setOnMouseEnter] = useState(false);
  const [chatbotOpen, setChatbotOpen] = useState(false);
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [mutedStates, setMutedStates] = useState({});
  const [feedbackMessage, setFeedbackMessage] = useState(null);
  const chatBodyRef = useRef(null);
  const fileInputRef = useRef(null);

  const { pathname } = useLocation();
  const account = useAppSelector((s) => s.authentication.account);
  const routes = transformMenuToRoutes([...(account?.menuMaster ?? [])]);

 
  function getUserId() {

  // Most likely place: your Redux "account" object

  return account?.id || account?.userId || ""; // Update this path as needed

}



function getClientId() {

  return account?.client_id || account?.clientId || ""; // Update as per actual key

}



function getSessionId() {

    // Always read from localStorage so it's persistent for the session/tab

    let sessionId = localStorage.getItem("session_id");

    if (!sessionId) {

        const userId = getUserId();

        if (userId) {

            sessionId = userId + "_" + Date.now();

            localStorage.setItem("session_id", sessionId);

        }

    }

    return sessionId || "";

}



const loadChatHistory = async () => {

    const userId = getUserId();

    const clientId = getClientId();

    const sessionId = getSessionId();

    

    if (!userId || !sessionId) {

        console.log("Cannot load history: missing userId or sessionId");

        return;

    }

    

    try {

        const jwt = Storage.local.get('jhi-authenticationToken') || 

                   Storage.session.get('jhi-authenticationToken');

        

        const response = await axios.get("http://localhost:5000/chat/history", {

            params: { user_id: userId, client_id: clientId, session_id: sessionId },

            headers: { 'Authorization': `Bearer ${jwt}` }

        });

        

        if (response.data.history && response.data.history.length > 0) {

            const formattedHistory = response.data.history.map(msg => ({

                sender: msg.sender,

                text: msg.text,

                timestamp: getCurrentTime(),

                feedbackGiven: msg.sender === "bot" ? false : undefined

            }));

            setMessages(formattedHistory);

        }

    } catch (error) {

        console.error("Error loading chat history:", error);

        // Don't show error to user - just start fresh

    }

};



  // Setup RTL cache

  useMemo(() => {

    const cacheRtl = createCache({

      key: "rtl",

      stylisPlugins: [rtlPlugin],

    });

    setRtlCache(cacheRtl);
  }, []);

  // scroll top on nav
  useEffect(() => {
    document.documentElement.scrollTop = 0;
    if (document.scrollingElement) document.scrollingElement.scrollTop = 0;
  }, [pathname]);

  // direction
  useEffect(() => {
    document.body.setAttribute('dir', direction);
  }, [direction]);

  // keep colors store synced
  useEffect(() => {
    const key = themeColor || getActivePaletteKey() || 'default';
    setActivePalette(key);
  }, [themeColor]);

  // Clear messages and add greeting when opening chatbot
  useEffect(() => {
    if (chatbotOpen) {
      setMessages([
        {
          sender: 'bot',
          text: 'Hello! How can I help you with your commission needs today?',
          timestamp: getCurrentTime(),
          feedbackGiven: false,
        },
      ]);
       setQuestion("");

      setMutedStates({});

      setFeedbackMessage(null);

    }

  }, [chatbotOpen]);



  // Load chat history when opening chatbot

  useEffect(() => {

    if (chatbotOpen) {

      setQuestion("");

      setMutedStates({});

      setFeedbackMessage(null);

      

      // Try to load existing history first

      loadChatHistory().then(() => {

        // Only add greeting if no history was loaded

        setMessages(prev => {

          if (prev.length === 0) {

            return [{

              sender: "bot",

              text: "Hello! How can I help you with your commission needs today?",

              timestamp: getCurrentTime(),

              feedbackGiven: false,

            }];

          }

          return prev;

        });

      });

    } else {

      setMessages([]);

      setQuestion("");
      setMutedStates({});
      setFeedbackMessage(null);
    }
  }, [chatbotOpen]);

  // Auto-scroll to bottom of chat when messages or feedback change
  useEffect(() => {
    if (chatBodyRef.current) {
      chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight;
    }
  }, [messages, feedbackMessage]);

  // compute drawerWidth
  const drawerWidth = useMemo(() => {
    if (sidenavClose) return 0;
    return miniSidenav ? 80 : 220;
  }, [miniSidenav, sidenavClose]);

  const handleConfiguratorOpen = () => {
    setOpenConfigurator(dispatchUI, !openConfigurator);
  };

  const handleOnMouseEnter = () => {
    if (miniSidenav && !onMouseEnter) {
      setMiniSidenav(dispatchUI, false);
      setOnMouseEnter(true);
    }
  };

  const handleOnMouseLeave = () => {
    if (onMouseEnter) {
      setMiniSidenav(dispatchUI, true);
      setOnMouseEnter(false);
    }
  };

  const handleChatbotToggle = () => {
    setChatbotOpen(!chatbotOpen);
  };

  const handleVolumeToggle = (index, text) => {
    setMutedStates((prev) => {
      const newMutedState = !prev[index];
      if (!newMutedState) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';
        window.speechSynthesis.speak(utterance);
      } else {
        window.speechSynthesis.cancel();
      }
      return { ...prev, [index]: newMutedState };
    });
  };

  const handleLike = (index) => {
    const timestamp = getCurrentTime();
    setMessages((prev) =>
      prev.map((msg, i) =>
        i === index ? { ...msg, feedbackGiven: true } : msg
      )
    );
    setFeedbackMessage({ text: 'Thank you for your feedback', timestamp, icon: 'thumb_up' });
    setTimeout(() => setFeedbackMessage(null), 2000);
  };

  const handleUnlike = (index) => {
    const timestamp = getCurrentTime();
    setMessages((prev) =>
      prev.map((msg, i) =>
        i === index ? { ...msg, feedbackGiven: true } : msg
      )
    );
    setFeedbackMessage({ text: 'Thank you for your feedback', timestamp, icon: 'thumb_down' });
    setTimeout(() => setFeedbackMessage(null), 2000);
  };

  const getCurrentTime = () => {
    const now = new Date();
    return now.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: true,
    });
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files: FileList | null = event.target.files;
    if (!files || files.length === 0) return;

    const formData = new FormData();
    Array.from(files).forEach((file: File) => {
      formData.append('files', file);
    });

    try {
      setIsLoading(true);
       const jwt =

        Storage.local.get('jhi-authenticationToken') ||

        Storage.session.get('jhi-authenticationToken');

      const response = await axios.post("http://localhost:5000/upload", formData, {

        headers: { 

                   "Authorization": `Bearer ${jwt}` 

                  },
      });
      setMessages((prev) => [
        ...prev,
        { sender: 'bot', text: response.data.response, timestamp: getCurrentTime() },
      ]);
    } catch (error) {
      console.error('File upload error:', error);
      const errorMessage = error.response?.data?.error
        ? `Error: ${error.response.data.error}`
        : `Error: Could not upload files. Please ensure the server is running and try again.`;
      setMessages((prev) => [
        ...prev,
        { sender: 'bot', text: errorMessage, timestamp: getCurrentTime() },
      ]);
    } finally {
      setIsLoading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleQuestionSubmit = async () => {
    if (!question.trim()) return;

    const userMessage = { sender: 'user', text: question, timestamp: getCurrentTime() };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion('');
    setIsLoading(true);

    /*let session_id = getSessionId();

  const user_id = getUserId();

  if (!session_id && user_id) {

    session_id = user_id + "_" + Date.now();

    localStorage.setItem("session_id", session_id);

  }*/

  const user_id = getUserId();

  let session_id = getSessionId();

  const client_id = getClientId();

   const requestBody = {

    user_id,

    client_id,

    session_id,

    message: question
   };

    try {
      const response = await axios.post('http://localhost:5000/chat', requestBody); // ✅ SENDING ALL FIELDS
      const botResponse = response.data.response || 'Sorry, I couldn\'t process your request.';
      setMessages((prev) => [
        ...prev,
        {
          sender: 'bot',
          text: response.data.formatted_response || response.data.response || "No response.",
          timestamp: getCurrentTime(),
          feedbackGiven: false,
        },
      ]);
    } catch (error) {
      const errorMessage = error.response?.data?.error
        ? `Error: ${error.response.data.error}`
        : `Error: Could not connect to the server. Please ensure the server is running and try again.`;
      setMessages((prev) => [
        ...prev,
        {
          sender: 'bot',
          text: errorMessage,
          timestamp: getCurrentTime(),
          feedbackGiven: false,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // const configsButton = (
  //   <MDBox
  //     display="flex"
  //     justifyContent="center"
  //     alignItems="center"
  //     width="3.25rem"
  //     height="3.25rem"
  //     bgColor="white"
  //     shadow="sm"
  //     borderRadius="50%"
  //     position="fixed"
  //     right="2rem"
  //     bottom="2rem"
  //     zIndex={1301}
  //     color="dark"
  //     sx={{ cursor: 'pointer' }}
  //     onClick={handleConfiguratorOpen}
  //   >
  //     <Icon fontSize="small" color="inherit">
  //       settings
  //     </Icon>
  //   </MDBox>
  // );

  const chatbotButton = (
    <Tooltip title="Chat with CommissionsBot" placement="top">
      <MDBox
        display="flex"
        justifyContent="center"
        alignItems="center"
        width="3rem"
        height="3rem"
        bgColor="#0052cc"
        shadow="md"
        borderRadius="50%"
        position="fixed"
        right="5.5rem"
        bottom="2rem"
        zIndex={1301}
        sx={{
          cursor: 'pointer',
          transition: 'all 0.3s ease-in-out',
          background: 'linear-gradient(135deg, #003087 0%, #00a1d6 100%)',
          '&:hover': {
            background: 'linear-gradient(135deg, #002060 0%, #007bb5 100%)',
            transform: 'scale(1.1)',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.2)',
          },
        }}
        onClick={handleChatbotToggle}
      >
        <Icon
          fontSize="medium"
          sx={{
            color: '#ffffff !important',
            transition: 'transform 0.3s ease-in-out',
            '&:hover': {
              transform: 'scale(1.2)',
            },
          }}
        >
          monetization_on
        </Icon>
      </MDBox>
    </Tooltip>
  );

  const appliedTheme = useMemo(() => {
    const factory = darkMode ? createDarkTheme : createLightTheme;
    const base = factory(themeColor || 'default');
    return { ...base, direction };
  }, [themeColor, darkMode, direction]);

  const ThemeWrapper = direction === 'rtl' && rtlCache ? CacheProvider : React.Fragment;
  const themeProps = direction === 'rtl' && rtlCache ? { value: rtlCache } : {};

  return (
    <ThemeWrapper {...themeProps}>
      <ThemeProvider theme={appliedTheme}>
        <CssBaseline />

        {/* Sidebar */}
        {routes.length > 0 && drawerWidth > 0 && (
          <MDBox
            sx={{
              position: 'fixed',
              left: 0,
              top: 0,
              bottom: 0,
              width: drawerWidth,
              transition: 'width 0.3s ease',
              overflow: 'hidden',
              zIndex: 10,
            }}
          >
            <Sidenav
              color={sidenavColor}
              brand={(transparentSidenav && !darkMode) || whiteSidenav ? brandDark : brandWhite}
              brandName="Creative Tim"
              routes={routes}
              // onMouseEnter={handleOnMouseEnter}
              // onMouseLeave={handleOnMouseLeave}
            />
          </MDBox>
        )}

        {/* Main content */}
        <MDBox
          sx={{
            ml: {
              xs: 0,
              xl: sidenavClose ? 0 : miniSidenav ? '80px' : '220px',
            },
            transition: 'margin-left 0.3s ease',
            width: {
              xs: '100%',
              xl: `calc(100% - ${sidenavClose ? 0 : miniSidenav ? '80px' : '220px'})`,
            },
            overflowX: 'hidden',
          }}
        >
          {layout === 'dashboard' && (
            <MDBox px={2}>
              <DashboardNavbar />
            </MDBox>
          )}

          <MDBox p={2} ml={2}>
            <Outlet />
          </MDBox>
        </MDBox>

        <Configurator />
        {/* {configsButton} */}
        {chatbotButton}

        {/* Chatbot Window */}
        {chatbotOpen &&(
        <MDBox
          sx={{
            position: 'fixed',
            bottom: chatbotOpen ? '6.5rem' : '2rem',
            right: '5.5rem',
            transform: chatbotOpen ? 'translateY(0) scale(1)' : 'translateY(100px) scale(0.5)',
            opacity: chatbotOpen ? 1 : 0,
            width: '360px',
            height: '500px',
            backgroundColor: appliedTheme.palette.background.paper,
            borderRadius: '20px',
            boxShadow: '0 10px 30px rgba(0, 0, 0, 0.2)',
            zIndex: 1000,
            flexDirection: 'column',
            display: 'flex',
            transition: 'transform 0.4s ease-out, opacity 0.4s ease-out, bottom 0.4s ease-out',
            transformOrigin: 'bottom center',
            border: `1px solid ${appliedTheme.palette.divider}`,
            '&:hover': {
              transform: chatbotOpen ? 'translateY(-5px) scale(1.02)' : 'translateY(100px) scale(0.5)',
            },
            '@keyframes bounceIn': {
              '0%': { transform: 'translateY(100px) scale(0.5)', opacity: 0 },
              '60%': { transform: 'translateY(-10px) scale(1.05)', opacity: 1 },
              '100%': { transform: 'translateY(0) scale(1)', opacity: 1 },
            },
            animation: chatbotOpen ? 'bounceIn 0.5s ease-out' : 'none',
          }}
        >
          <MDBox
            sx={{
              background: 'linear-gradient(135deg, #003087 0%, #00a1d6 100%)',
              color: 'white',
              padding: '16px',
              borderRadius: '20px 20px 0 0',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <Box display="flex" alignItems="center" gap={1}>
              <Avatar sx={{ bgcolor: 'white', color: '#003087' }}>
                <Icon>monetization_on</Icon>
              </Avatar>
              <Typography
                variant="h6"
                sx={{
                  color: '#ffffff !important',
                  zIndex: 2,
                  position: 'relative',
                }}
              >
                CommissionsBot
              </Typography>
            </Box>
            <Box>
              <IconButton onClick={handleChatbotToggle}>
                <Icon sx={{ color: 'white' }}>close</Icon>
              </IconButton>
            </Box>
          </MDBox>
          <MDBox
            ref={chatBodyRef}
            sx={{
              flex: 1,
              padding: '16px',
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: '24px',
              backgroundColor: appliedTheme.palette.background.default,
              position: 'relative',
            }}
          >
            {messages.map((msg, index) => (
              <MDBox
                key={index}
                sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                  gap: '4px',
                }}
              >
                <Box
                  sx={{
                    display: 'flex',
                    flexDirection: msg.sender === 'user' ? 'row-reverse' : 'row',
                    alignItems: 'flex-start',
                    gap: '8px',
                    width: '100%',
                    position: 'relative',
                  }}
                >
                  <Avatar
                    sx={{
                      bgcolor:
                        msg.sender === 'user'
                          ? '#003087'
                          : appliedTheme.palette.secondary.main,
                      width: 24,
                      height: 24,
                      fontSize: '0.8rem',
                      order: msg.sender === 'user' ? 1 : 0,
                      marginLeft: msg.sender === 'user' ? 0 : '8px',
                      marginRight: msg.sender === 'user' ? '8px' : 0,
                    }}
                  >
                    {msg.sender === 'user' ? <Icon fontSize="small">person</Icon> : <Icon fontSize="small">monetization_on</Icon>}
                  </Avatar>
                  <MDBox
                    sx={{
                      backgroundColor:
                        msg.sender === 'user'
                          ? '#003087'
                          : appliedTheme.palette.grey[200],
                      color: msg.sender === 'user' ? '#FFFFFF' : 'text.primary',
                      borderRadius: msg.sender === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
                      padding: '12px 16px',
                      maxWidth: '70%',
                      wordBreak: 'break-word',
                      boxShadow: '0 2px 6px rgba(0, 0, 0, 0.1)',
                      transition: 'all 0.2s',
                      '&:hover': { transform: 'translateY(-2px)' },
                      position: 'relative',
                    }}
                  >
                    <Typography
                      variant="body2"
                      sx={{
                        color: msg.sender === 'bot' ? 'rgb(42, 65, 157)' : undefined,
                      }}
                    >
                      <ReactMarkdown>{msg.text}</ReactMarkdown>
                    </Typography>
                    {msg.sender === 'bot' && (
                      <IconButton
                        size="small"
                        onClick={() => handleVolumeToggle(index, msg.text)}
                        sx={{
                          position: 'absolute',
                          right: 0,
                          top: '50%',
                          transform: 'translate(50%, -50%)',
                          width: '20px',
                          height: '20px',
                          backgroundColor: '#1976d2',
                          border: `1px solid ${appliedTheme.palette.divider}`,
                          borderRadius: '50%',
                          '&:hover': {
                            backgroundColor: '#1565c0',
                            transform: 'translate(50%, -50%) scale(1.2)',
                          },
                          transition: 'transform 0.2s',
                          zIndex: 1,
                        }}
                      >
                        <Icon
                          fontSize="small"
                          sx={{
                            fontSize: '0.9rem',
                            color: '#fff !important',
                            transition: 'none',
                          }}
                        >
                          {mutedStates[index] ? 'volume_off' : 'volume_up'}
                        </Icon>
                      </IconButton>
                    )}
                    <Typography
                      variant="caption"
                      color="#000000"
                      sx={{
                        opacity: 0.8,
                        fontSize: '0.7rem',
                        position: 'absolute',
                        bottom: '-18px',
                        left: msg.sender === 'user' ? 'auto' : 0,
                        right: msg.sender === 'user' ? 0 : 'auto',
                      }}
                    >
                      {msg.timestamp}
                    </Typography>
                    {msg.sender === 'bot' && !msg.feedbackGiven && (
                      <Box
                        sx={{
                          position: 'absolute',
                          bottom: '-18px',
                          right: 0,
                          display: 'flex',
                          gap: '4px',
                          alignItems: 'center',
                        }}
                      >
                        <Icon
                          onClick={() => handleLike(index)}
                          sx={{
                            fontSize: '14px',
                            cursor: 'pointer',
                            color: appliedTheme.palette.grey[600],
                            transition: 'color 0.2s',
                            '&:hover': {
                              color: appliedTheme.palette.primary.main,
                            },
                          }}
                          aria-label="like"
                        >
                          thumb_up
                        </Icon>
                        <Icon
                          onClick={() => handleUnlike(index)}
                          sx={{
                            fontSize: '14px',
                            cursor: 'pointer',
                            color: appliedTheme.palette.grey[600],
                            transition: 'color 0.2s',
                            '&:hover': {
                              color: '#f44336',
                            },
                          }}
                          aria-label="dislike"
                        >
                          thumb_down
                        </Icon>
                      </Box>
                    )}
                  </MDBox>
                </Box>
              </MDBox>
            ))}
            {feedbackMessage && (
              <MDBox
                sx={{
                  backgroundColor:
                    feedbackMessage.icon === 'thumb_up' ? '#e3f2fd' : '#ffebee',
                  borderRadius: '50px',
                  padding: '12px 20px',
                  boxShadow: '0 6px 20px rgba(0,0,0,0.15)',
                  textAlign: 'center',
                  alignSelf: 'center',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  animation: 'fadeInOut 2s ease-in-out',
                  '@keyframes fadeInOut': {
                    '0%': { opacity: 0, transform: 'scale(0.8)' },
                    '10%': { opacity: 1, transform: 'scale(1)' },
                    '90%': { opacity: 1, transform: 'scale(1)' },
                    '100%': { opacity: 0, transform: 'scale(0.8)' },
                  },
                }}
              >
                <Avatar
                  sx={{
                    bgcolor: feedbackMessage.icon === 'thumb_up' ? '#1976d2' : '#d32f2f',
                    width: 40,
                    height: 40,
                  }}
                >
                  <Icon sx={{ color: 'white', fontSize: '24px' }}>{feedbackMessage.icon}</Icon>
                </Avatar>
                <Box>
                  <Typography variant="body1" sx={{ fontWeight: 500, color: '#000' }}>
                    {feedbackMessage.text}
                  </Typography>
                  <Typography
                    variant="caption"
                    sx={{ color: '#444', fontSize: '0.75rem' }}
                  >
                    {feedbackMessage.timestamp}
                  </Typography>
                </Box>
              </MDBox>
            )}
            {isLoading && (
              <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>
                Loading...
              </Typography>
            )}
          </MDBox>
          <MDBox
            sx={{
              padding: '12px',
              borderTop: `1px solid ${appliedTheme.palette.divider}`,
              backgroundColor: appliedTheme.palette.background.paper,
              display: 'flex',
              alignItems: 'center',
              borderRadius: '0 0 20px 20px',
            }}
          >
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Type your question..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter') handleQuestionSubmit();
              }}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: '24px',
                  backgroundColor: appliedTheme.palette.grey[100],
                  '& fieldset': { border: 'none' },
                },
                '& .MuiInputBase-input': { padding: '10px 16px' },
              }}
            />
            <Tooltip title="Upload Documents">
              <IconButton
                sx={{
                  marginLeft: '8px',
                  color: appliedTheme.palette.primary.main,
                  '&:hover': { bgcolor: appliedTheme.palette.primary.light },
                }}
                disabled={isLoading}
                onClick={() => fileInputRef.current?.click()}
              >
                <Icon>upload</Icon>
              </IconButton>
            </Tooltip>
            <input
              type="file"
              multiple
              accept=".pdf,.docx,.txt,.json,.xls,.xlsx,.png,.jpg,.jpeg,.tiff"
              style={{ display: 'none' }}
              id="file-upload"
              ref={fileInputRef}
              onChange={handleFileUpload}
            />
            <IconButton
              onClick={handleQuestionSubmit}
              sx={{
                marginLeft: '8px',
                color: appliedTheme.palette.primary.main,
                '&:hover': { bgcolor: appliedTheme.palette.primary.light },
              }}
              disabled={isLoading}
            >
              <Icon>send</Icon>
            </IconButton>
          </MDBox>
        </MDBox>
        )}
      </ThemeProvider>
    </ThemeWrapper>
  );
};

export default FullLayout;