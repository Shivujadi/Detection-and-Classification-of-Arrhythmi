/**
 * PulseNews - Real-Time News & Audio Voice Reader Application
 * JavaScript Application Architecture
 */

// ==========================================================================
// 1. Application State & Storage
// ==========================================================================
const AppState = {
    category: 'general',
    country: 'in',
    query: '',
    sortBy: 'newest',
    articles: [],
    savedArticles: JSON.parse(localStorage.getItem('pulsenews_bookmarks') || '[]'),
    theme: localStorage.getItem('pulsenews_theme') || 'dark',
    apiKey: localStorage.getItem('pulsenews_apikey') || '',
    apiProvider: localStorage.getItem('pulsenews_provider') || 'auto',
    viewingBookmarks: false,
    currentSpeakingArticleId: null
};

// High-Quality Fallback / Demo Dataset (Ensures 100% functionality offline/error)
const FallbackNewsData = {
    general: [
        {
            id: 'demo-1',
            title: 'Global Tech & Innovation Summit 2026 Announces Next-Gen AI Breakthroughs',
            description: 'Industry leaders from around the globe gather to discuss quantum computing, ethical artificial intelligence, and autonomous robotics shaping the future of global technology.',
            source: { name: 'TechCrunch' },
            publishedAt: new Date(Date.now() - 3600000 * 2).toISOString(),
            urlToImage: 'https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1000&q=80',
            url: 'https://techcrunch.com'
        },
        {
            id: 'demo-2',
            title: 'Renewable Energy Milestones: Solar & Wind Power Supply Over 60% of Grid Power',
            description: 'A landmark achievement in sustainable clean energy as massive solar arrays and offshore wind farms record record-breaking electricity generation worldwide.',
            source: { name: 'BBC News' },
            publishedAt: new Date(Date.now() - 3600000 * 5).toISOString(),
            urlToImage: 'https://images.unsplash.com/photo-1466611653911-95081537e5b7?auto=format&fit=crop&w=1000&q=80',
            url: 'https://bbc.com/news'
        },
        {
            id: 'demo-3',
            title: 'Deep Space Mission Transmits High-Resolution Images of Distant Exoplanets',
            description: 'Astronomers celebrate as the latest space observatory sends stunning spectroscopic data hinting at potentially atmospheric water vapor on alien worlds.',
            source: { name: 'National Geographic' },
            publishedAt: new Date(Date.now() - 3600000 * 12).toISOString(),
            urlToImage: 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=1000&q=80',
            url: 'https://nationalgeographic.com'
        },
        {
            id: 'demo-4',
            title: 'Global Economic Report Predicts Strong Growth in Emerging Digital Markets',
            description: 'Financial analysts highlight key market trends showing rapid digital infrastructure adoption and expanding fintech services across international economies.',
            source: { name: 'Bloomberg' },
            publishedAt: new Date(Date.now() - 3600000 * 24).toISOString(),
            urlToImage: 'https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&w=1000&q=80',
            url: 'https://bloomberg.com'
        }
    ]
};

// Default Placeholder Image for broken article links
const DEFAULT_ARTICLE_IMAGE = 'https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&w=1000&q=80';

// ==========================================================================
// 2. Text-To-Speech (TTS) Engine
// ==========================================================================
class TTSEngine {
    constructor() {
        this.synth = window.speechSynthesis;
        this.utterance = null;
        this.voices = [];
        this.isPlaying = false;
        this.isPaused = false;
        this.currentArticle = null;
        this.rate = 1.0;
        this.selectedVoiceIndex = 0;

        this.initVoices();
    }

    initVoices() {
        if (!this.synth) return;

        const populate = () => {
            this.voices = this.synth.getVoices().filter(v => v.lang.startsWith('en'));
            const voiceSelect = document.getElementById('ttsVoiceSelect');
            if (!voiceSelect) return;

            voiceSelect.innerHTML = '';
            this.voices.forEach((voice, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `${voice.name} (${voice.lang})`;
                if (voice.default) option.selected = true;
                voiceSelect.appendChild(option);
            });
        };

        populate();
        if (speechSynthesis.onvoiceschanged !== undefined) {
            speechSynthesis.onvoiceschanged = populate;
        }
    }

    speak(article, cardElement) {
        if (!this.synth) {
            showToast('Text-to-speech is not supported in your browser.', 'error');
            return;
        }

        // Stop existing speech if playing
        this.stop();

        this.currentArticle = article;
        AppState.currentSpeakingArticleId = article.id || article.title;

        // Construct clean text snippet for speech output
        const textToRead = `Article headline: ${article.title}. From ${article.source?.name || 'News Source'}. ${article.description || ''}`;

        this.utterance = new SpeechSynthesisUtterance(textToRead);
        this.utterance.rate = this.rate;

        if (this.voices[this.selectedVoiceIndex]) {
            this.utterance.voice = this.voices[this.selectedVoiceIndex];
        }

        this.utterance.onstart = () => {
            this.isPlaying = true;
            this.isPaused = false;
            this.updatePlayerUI(article.title, true);
            this.highlightActiveCard(cardElement);
        };

        this.utterance.onend = () => {
            this.resetState();
        };

        this.utterance.onerror = (e) => {
            console.error('Speech synthesis error:', e);
            this.resetState();
            showToast('Speech playback encountered an issue.', 'error');
        };

        this.synth.speak(this.utterance);
    }

    togglePlayPause() {
        if (!this.synth || !this.currentArticle) return;

        if (this.isPlaying && !this.isPaused) {
            this.synth.pause();
            this.isPaused = true;
            this.updatePlayerUI(this.currentArticle.title, false);
        } else if (this.isPaused) {
            this.synth.resume();
            this.isPaused = false;
            this.updatePlayerUI(this.currentArticle.title, true);
        }
    }

    stop() {
        if (!this.synth) return;
        this.synth.cancel();
        this.resetState();
    }

    setRate(rate) {
        this.rate = parseFloat(rate);
        if (this.isPlaying && this.currentArticle) {
            // Re-speak with new speed
            this.speak(this.currentArticle);
        }
    }

    setVoiceIndex(index) {
        this.selectedVoiceIndex = parseInt(index, 10);
        if (this.isPlaying && this.currentArticle) {
            this.speak(this.currentArticle);
        }
    }

    resetState() {
        this.isPlaying = false;
        this.isPaused = false;
        this.currentArticle = null;
        AppState.currentSpeakingArticleId = null;

        const playerBar = document.getElementById('ttsPlayerBar');
        if (playerBar) playerBar.style.display = 'none';

        // Clear highlighted cards
        document.querySelectorAll('.news-card.speaking-active').forEach(el => el.classList.remove('speaking-active'));
    }

    updatePlayerUI(title, playing) {
        const playerBar = document.getElementById('ttsPlayerBar');
        const currentTitleEl = document.getElementById('ttsCurrentTitle');
        const playIcon = document.querySelector('#ttsPlayPauseBtn .play-icon');
        const pauseIcon = document.querySelector('#ttsPlayPauseBtn .pause-icon');
        const visualizer = document.getElementById('ttsVisualizer');

        if (!playerBar) return;

        playerBar.style.display = 'block';
        if (currentTitleEl) currentTitleEl.textContent = title;

        if (playing) {
            if (playIcon) playIcon.style.display = 'none';
            if (pauseIcon) pauseIcon.style.display = 'block';
            if (visualizer) visualizer.classList.remove('paused');
        } else {
            if (playIcon) playIcon.style.display = 'block';
            if (pauseIcon) pauseIcon.style.display = 'none';
            if (visualizer) visualizer.classList.add('paused');
        }
    }

    highlightActiveCard(cardElement) {
        document.querySelectorAll('.news-card.speaking-active').forEach(el => el.classList.remove('speaking-active'));
        if (cardElement) {
            cardElement.classList.add('speaking-active');
            cardElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }
}

const tts = new TTSEngine();

// ==========================================================================
// 3. API Data Fetching Layer
// ==========================================================================
async function fetchNewsArticles() {
    showSkeletonLoader(true);
    hideEmptyState();

    try {
        let articles = [];

        // Check if viewing saved bookmarks
        if (AppState.viewingBookmarks) {
            articles = AppState.savedArticles;
            renderArticles(articles);
            return;
        }

        // Custom API key check
        if (AppState.apiKey && (AppState.apiProvider === 'newsapi' || AppState.apiProvider === 'auto')) {
            try {
                let url = '';
                if (AppState.query) {
                    url = `https://newsapi.org/v2/everything?q=${encodeURIComponent(AppState.query)}&apiKey=${AppState.apiKey}&pageSize=30`;
                } else {
                    url = `https://newsapi.org/v2/top-headlines?category=${AppState.category}&country=${AppState.country}&apiKey=${AppState.apiKey}&pageSize=30`;
                }
                const res = await fetch(url);
                const data = await res.json();
                if (data.status === 'ok' && data.articles && data.articles.length > 0) {
                    articles = data.articles;
                }
            } catch (err) {
                console.warn('NewsAPI.org request failed, switching to public mirror:', err);
            }
        }

        // Primary Public CORS Mirror (SauravTech API mirror)
        if (articles.length === 0) {
            try {
                let url = '';
                if (AppState.query) {
                    // Fetch headlines and client-side filter for search query
                    url = `https://saurav.tech/NewsAPI/top-headlines/category/${AppState.category}/${AppState.country}.json`;
                } else {
                    url = `https://saurav.tech/NewsAPI/top-headlines/category/${AppState.category}/${AppState.country}.json`;
                }

                const response = await fetch(url);
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                const data = await response.json();
                
                if (data.articles && data.articles.length > 0) {
                    articles = data.articles;
                    // Client-side query search filter
                    if (AppState.query) {
                        const q = AppState.query.toLowerCase();
                        articles = articles.filter(a => 
                            (a.title && a.title.toLowerCase().includes(q)) || 
                            (a.description && a.description.toLowerCase().includes(q))
                        );
                    }
                }
            } catch (error) {
                console.warn('Primary public API failed, trying secondary fallback:', error);
            }
        }

        // Secondary fallback API (Spaceflight News / Open News Endpoint if category is science/tech)
        if (articles.length === 0 && (AppState.category === 'technology' || AppState.category === 'science')) {
            try {
                const res = await fetch('https://api.spaceflightnewsapi.net/v4/articles/?limit=20');
                const data = await res.json();
                if (data.results && data.results.length > 0) {
                    articles = data.results.map(item => ({
                        id: `space-${item.id}`,
                        title: item.title,
                        description: item.summary,
                        source: { name: item.news_site || 'SpaceNews' },
                        publishedAt: item.published_at,
                        urlToImage: item.image_url,
                        url: item.url
                    }));
                }
            } catch (err) {
                console.warn('Secondary fallback failed:', err);
            }
        }

        // Offline / Fallback Demo Data if network or all endpoints fail
        if (articles.length === 0) {
            console.info('Using high-quality fallback demo dataset.');
            articles = FallbackNewsData[AppState.category] || FallbackNewsData.general;
            showToast('Loaded demo articles (offline/fallback mode).', 'info');
        }

        // Format & Assign Articles
        AppState.articles = articles.map((article, idx) => ({
            id: article.id || `art-${idx}-${Date.now()}`,
            title: article.title || 'Untitled Headline',
            description: article.description || 'No detailed description available for this news story.',
            source: article.source || { name: 'News Source' },
            publishedAt: article.publishedAt || article.published_at || new Date().toISOString(),
            urlToImage: article.urlToImage || article.image_url || DEFAULT_ARTICLE_IMAGE,
            url: article.url || '#'
        }));

        sortAndRenderArticles();

    } catch (err) {
        console.error('Failed to fetch news articles:', err);
        showEmptyState('Error Loading News', 'Unable to load news articles at this moment. Please check your internet connection or try again later.');
    } finally {
        showSkeletonLoader(false);
    }
}

function sortAndRenderArticles() {
    let sorted = [...AppState.articles];
    if (AppState.sortBy === 'newest') {
        sorted.sort((a, b) => new Date(b.publishedAt) - new Date(a.publishedAt));
    } else if (AppState.sortBy === 'oldest') {
        sorted.sort((a, b) => new Date(a.publishedAt) - new Date(b.publishedAt));
    }

    renderArticles(sorted);
}

// ==========================================================================
// 4. UI Rendering Functions
// ==========================================================================
function renderArticles(articles) {
    const grid = document.getElementById('newsGrid');
    const hero = document.getElementById('heroContainer');
    const countPill = document.getElementById('articleCountPill');

    if (!grid) return;

    grid.innerHTML = '';
    
    if (articles.length === 0) {
        showEmptyState('No News Articles Found', 'Try adjusting your search query or switching to another category.');
        if (hero) hero.style.display = 'none';
        if (countPill) countPill.textContent = '0 Articles';
        return;
    }

    hideEmptyState();

    if (countPill) countPill.textContent = `${articles.length} Article${articles.length > 1 ? 's' : ''}`;

    // Separate top featured article for Hero Section if not searching
    let gridArticles = articles;
    if (!AppState.query && articles.length > 1 && !AppState.viewingBookmarks && hero) {
        const topHeroArticle = articles[0];
        gridArticles = articles.slice(1);
        renderHeroArticle(topHeroArticle);
        hero.style.display = 'block';
    } else if (hero) {
        hero.style.display = 'none';
    }

    // Render Grid Cards using document fragment for smooth performance
    const fragment = document.createDocumentFragment();

    gridArticles.forEach((article) => {
        const card = createNewsCard(article);
        fragment.appendChild(card);
    });

    grid.appendChild(fragment);

    // Update Top Ticker text with breaking news title
    const tickerText = document.getElementById('tickerText');
    if (tickerText && articles.length > 0) {
        tickerText.textContent = `TRENDING: ${articles[0].title}`;
    }
}

function renderHeroArticle(article) {
    const heroContainer = document.getElementById('heroContainer');
    if (!heroContainer) return;

    const isBookmarked = isArticleSaved(article);

    heroContainer.innerHTML = `
        <article class="hero-card">
            <div class="hero-img-wrapper">
                <img src="${escapeHtml(article.urlToImage)}" alt="${escapeHtml(article.title)}" onerror="this.src='${DEFAULT_ARTICLE_IMAGE}'">
                <span class="hero-badge">FEATURED</span>
            </div>
            <div class="hero-content">
                <div>
                    <div class="hero-meta">
                        <span class="source-tag">${escapeHtml(article.source?.name || 'Top News')}</span>
                        <span>•</span>
                        <span>${formatRelativeTime(article.publishedAt)}</span>
                    </div>
                    <h3 class="hero-title">${escapeHtml(article.title)}</h3>
                    <p class="hero-description">${escapeHtml(article.description)}</p>
                </div>
                <div class="hero-actions">
                    <a href="${escapeHtml(article.url)}" target="_blank" rel="noopener noreferrer" class="primary-btn">
                        Read Full Article
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="7" y1="17" x2="17" y2="7"></line>
                            <polyline points="7 7 17 7 17 17"></polyline>
                        </svg>
                    </a>
                    <button class="secondary-btn hero-speak-btn">
                        🔊 Listen News
                    </button>
                    <button class="secondary-btn hero-bookmark-btn ${isBookmarked ? 'saved' : ''}">
                        ${isBookmarked ? '❤️ Saved' : '🔖 Save'}
                    </button>
                </div>
            </div>
        </article>
    `;

    // Attach hero events
    const speakBtn = heroContainer.querySelector('.hero-speak-btn');
    if (speakBtn) {
        speakBtn.addEventListener('click', () => {
            tts.speak(article, heroContainer.querySelector('.hero-card'));
        });
    }

    const bookmarkBtn = heroContainer.querySelector('.hero-bookmark-btn');
    if (bookmarkBtn) {
        bookmarkBtn.addEventListener('click', () => {
            toggleBookmark(article);
            renderArticles(AppState.articles);
        });
    }
}

function createNewsCard(article) {
    const card = document.createElement('article');
    card.className = `news-card ${AppState.currentSpeakingArticleId === article.id ? 'speaking-active' : ''}`;
    card.dataset.id = article.id;

    const isBookmarked = isArticleSaved(article);

    card.innerHTML = `
        <div class="card-img-wrapper">
            <img src="${escapeHtml(article.urlToImage)}" alt="${escapeHtml(article.title)}" onerror="this.src='${DEFAULT_ARTICLE_IMAGE}'" loading="lazy">
            <span class="card-source-badge">${escapeHtml(article.source?.name || 'News')}</span>
            <button class="card-bookmark-btn ${isBookmarked ? 'saved' : ''}" title="${isBookmarked ? 'Remove Bookmark' : 'Save Article'}" aria-label="Save Article">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="${isBookmarked ? 'currentColor' : 'none'}" stroke="currentColor" stroke-width="2">
                    <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"></path>
                </svg>
            </button>
        </div>
        <div class="card-body">
            <div class="card-date">${formatRelativeTime(article.publishedAt)}</div>
            <h3 class="card-title">${escapeHtml(article.title)}</h3>
            <p class="card-description">${escapeHtml(article.description)}</p>
            <div class="card-footer">
                <a href="${escapeHtml(article.url)}" target="_blank" rel="noopener noreferrer" class="read-more-btn">
                    Read More ↗
                </a>
                <button class="speech-btn card-speech-btn" title="Listen Article">
                    🔊 Listen
                </button>
            </div>
        </div>
    `;

    // Add card action handlers
    const speechBtn = card.querySelector('.card-speech-btn');
    if (speechBtn) {
        speechBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            tts.speak(article, card);
        });
    }

    const bookmarkBtn = card.querySelector('.card-bookmark-btn');
    if (bookmarkBtn) {
        bookmarkBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleBookmark(article);
            card.querySelector('.card-bookmark-btn').classList.toggle('saved');
            const svg = card.querySelector('.card-bookmark-btn svg');
            svg.setAttribute('fill', isArticleSaved(article) ? 'currentColor' : 'none');
        });
    }

    return card;
}

// ==========================================================================
// 5. Utility & Helper Functions
// ==========================================================================
function formatRelativeTime(dateString) {
    if (!dateString) return 'Recently';
    const date = new Date(dateString);
    if (isNaN(date.getTime())) return 'Recently';

    const now = new Date();
    const diffInSeconds = Math.floor((now - date) / 1000);

    if (diffInSeconds < 60) return 'Just now';
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`;
    if (diffInSeconds < 172800) return 'Yesterday';
    
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>"']/g, function (m) {
        return {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        }[m];
    });
}

function showSkeletonLoader(show) {
    const skeleton = document.getElementById('newsSkeleton');
    const grid = document.getElementById('newsGrid');
    if (skeleton) skeleton.style.display = show ? 'grid' : 'none';
    if (grid && show) grid.innerHTML = '';
}

function showEmptyState(title, message) {
    const emptyState = document.getElementById('emptyState');
    const titleEl = document.getElementById('emptyTitle');
    const msgEl = document.getElementById('emptyMessage');
    const grid = document.getElementById('newsGrid');

    if (grid) grid.innerHTML = '';
    if (titleEl) titleEl.textContent = title;
    if (msgEl) msgEl.textContent = message;
    if (emptyState) emptyState.style.display = 'block';
}

function hideEmptyState() {
    const emptyState = document.getElementById('emptyState');
    if (emptyState) emptyState.style.display = 'none';
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span>${type === 'error' ? '⚠️' : 'ℹ️'}</span>
        <span>${escapeHtml(message)}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// ==========================================================================
// 6. Bookmark Management
// ==========================================================================
function isArticleSaved(article) {
    return AppState.savedArticles.some(saved => saved.title === article.title);
}

function toggleBookmark(article) {
    const index = AppState.savedArticles.findIndex(saved => saved.title === article.title);
    if (index > -1) {
        AppState.savedArticles.splice(index, 1);
        showToast('Article removed from saved bookmarks.', 'info');
    } else {
        AppState.savedArticles.push(article);
        showToast('Article saved to bookmarks!', 'info');
    }

    localStorage.setItem('pulsenews_bookmarks', JSON.stringify(AppState.savedArticles));
    updateBookmarkBadge();
}

function updateBookmarkBadge() {
    const badge = document.getElementById('bookmarkCount');
    if (badge) badge.textContent = AppState.savedArticles.length;
}

// ==========================================================================
// 7. Event Listeners & Initialization
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Theme
    document.documentElement.setAttribute('data-theme', AppState.theme);
    const themeBtn = document.getElementById('themeToggleBtn');
    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            AppState.theme = AppState.theme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', AppState.theme);
            localStorage.setItem('pulsenews_theme', AppState.theme);
        });
    }

    // 2. Initialize Live Clock
    const clockEl = document.getElementById('liveClock');
    const updateClock = () => {
        if (clockEl) {
            const now = new Date();
            clockEl.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }
    };
    updateClock();
    setInterval(updateClock, 1000);

    // 3. Category Tabs
    const categoryTabs = document.getElementById('categoryTabs');
    if (categoryTabs) {
        categoryTabs.addEventListener('click', (e) => {
            const btn = e.target.closest('.category-btn');
            if (!btn) return;

            document.querySelectorAll('.category-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            AppState.category = btn.dataset.category;
            AppState.viewingBookmarks = false;
            AppState.query = '';

            const searchInput = document.getElementById('searchInput');
            if (searchInput) searchInput.value = '';
            document.getElementById('clearSearchBtn').hidden = true;

            const categoryHeading = document.getElementById('currentCategoryHeading');
            if (categoryHeading) categoryHeading.textContent = `${btn.textContent.trim()} Headlines`;

            fetchNewsArticles();
        });
    }

    // 4. Country Selector
    const countrySelect = document.getElementById('countrySelect');
    if (countrySelect) {
        countrySelect.value = AppState.country;
        countrySelect.addEventListener('change', (e) => {
            AppState.country = e.target.value;
            fetchNewsArticles();
        });
    }

    // 5. Search Bar & Debounce
    const searchInput = document.getElementById('searchInput');
    const clearSearchBtn = document.getElementById('clearSearchBtn');
    const searchSubmitBtn = document.getElementById('searchSubmitBtn');
    let searchDebounceTimer = null;

    const performSearch = () => {
        const val = searchInput.value.trim();
        AppState.query = val;
        AppState.viewingBookmarks = false;
        clearSearchBtn.hidden = !val;

        const categoryHeading = document.getElementById('currentCategoryHeading');
        if (categoryHeading) {
            categoryHeading.textContent = val ? `Search results for "${val}"` : `${AppState.category.toUpperCase()} Headlines`;
        }

        fetchNewsArticles();
    };

    if (searchInput) {
        searchInput.addEventListener('input', () => {
            clearTimeout(searchDebounceTimer);
            clearSearchBtn.hidden = !searchInput.value;
            searchDebounceTimer = setTimeout(performSearch, 450);
        });

        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                clearTimeout(searchDebounceTimer);
                performSearch();
            }
        });
    }

    if (clearSearchBtn) {
        clearSearchBtn.addEventListener('click', () => {
            searchInput.value = '';
            clearSearchBtn.hidden = true;
            performSearch();
        });
    }

    if (searchSubmitBtn) {
        searchSubmitBtn.addEventListener('click', performSearch);
    }

    // 6. Sort Select
    const sortSelect = document.getElementById('sortSelect');
    if (sortSelect) {
        sortSelect.addEventListener('change', (e) => {
            AppState.sortBy = e.target.value;
            sortAndRenderArticles();
        });
    }

    // 7. Refresh Button
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            fetchNewsArticles();
            showToast('News feed refreshed.', 'info');
        });
    }

    // 8. Bookmarks Toggle Button
    const bookmarksBtn = document.getElementById('bookmarksToggleBtn');
    if (bookmarksBtn) {
        bookmarksBtn.addEventListener('click', () => {
            AppState.viewingBookmarks = !AppState.viewingBookmarks;
            const categoryHeading = document.getElementById('currentCategoryHeading');

            if (AppState.viewingBookmarks) {
                if (categoryHeading) categoryHeading.textContent = '❤️ Saved Articles';
                renderArticles(AppState.savedArticles);
            } else {
                if (categoryHeading) categoryHeading.textContent = `${AppState.category.toUpperCase()} Headlines`;
                fetchNewsArticles();
            }
        });
    }

    // 9. Reset Search & Filters Button
    const emptyResetBtn = document.getElementById('emptyResetBtn');
    if (emptyResetBtn) {
        emptyResetBtn.addEventListener('click', () => {
            AppState.query = '';
            AppState.category = 'general';
            AppState.viewingBookmarks = false;
            if (searchInput) searchInput.value = '';
            if (clearSearchBtn) clearSearchBtn.hidden = true;

            document.querySelectorAll('.category-btn').forEach(b => {
                b.classList.toggle('active', b.dataset.category === 'general');
            });

            fetchNewsArticles();
        });
    }

    // 10. TTS Player Controls
    const ttsPlayPauseBtn = document.getElementById('ttsPlayPauseBtn');
    const ttsStopBtn = document.getElementById('ttsStopBtn');
    const ttsCloseBtn = document.getElementById('ttsCloseBtn');
    const ttsVoiceSelect = document.getElementById('ttsVoiceSelect');
    const ttsRateSelect = document.getElementById('ttsRateSelect');

    if (ttsPlayPauseBtn) ttsPlayPauseBtn.addEventListener('click', () => tts.togglePlayPause());
    if (ttsStopBtn) ttsStopBtn.addEventListener('click', () => tts.stop());
    if (ttsCloseBtn) ttsCloseBtn.addEventListener('click', () => tts.stop());

    if (ttsVoiceSelect) {
        ttsVoiceSelect.addEventListener('change', (e) => tts.setVoiceIndex(e.target.value));
    }

    if (ttsRateSelect) {
        ttsRateSelect.addEventListener('change', (e) => tts.setRate(e.target.value));
    }

    // 11. Modal Settings System
    const modal = document.getElementById('settingsModal');
    const settingsBtn = document.getElementById('settingsModalBtn');
    const modalCloseBtn = document.getElementById('modalCloseBtn');
    const modalCancelBtn = document.getElementById('modalCancelBtn');
    const modalSaveBtn = document.getElementById('modalSaveBtn');
    const apiKeyInput = document.getElementById('apiKeyInput');
    const apiProviderSelect = document.getElementById('apiProviderSelect');

    if (settingsBtn && modal) {
        settingsBtn.addEventListener('click', () => {
            if (apiKeyInput) apiKeyInput.value = AppState.apiKey;
            if (apiProviderSelect) apiProviderSelect.value = AppState.apiProvider;
            modal.style.display = 'flex';
        });
    }

    const closeModal = () => {
        if (modal) modal.style.display = 'none';
    };

    if (modalCloseBtn) modalCloseBtn.addEventListener('click', closeModal);
    if (modalCancelBtn) modalCancelBtn.addEventListener('click', closeModal);

    if (modalSaveBtn) {
        modalSaveBtn.addEventListener('click', () => {
            AppState.apiKey = apiKeyInput.value.trim();
            AppState.apiProvider = apiProviderSelect.value;
            localStorage.setItem('pulsenews_apikey', AppState.apiKey);
            localStorage.setItem('pulsenews_provider', AppState.apiProvider);
            closeModal();
            showToast('Settings saved successfully.', 'info');
            fetchNewsArticles();
        });
    }

    // Logo Click Handler (Go back home)
    const logoBtn = document.getElementById('logoBtn');
    if (logoBtn) {
        logoBtn.addEventListener('click', () => {
            AppState.category = 'general';
            AppState.query = '';
            AppState.viewingBookmarks = false;
            fetchNewsArticles();
        });
    }

    updateBookmarkBadge();

    // Initial Article Fetch
    fetchNewsArticles();
});
