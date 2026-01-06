#!/usr/bin/env python3

import base64
import os
import sys
import time
import random
import re
import json
from dataclasses import dataclass, asdict
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional, Set, Generator
from urllib.parse import urljoin, urlparse, urldefrag, quote
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

import requests
from PIL import Image, UnidentifiedImageError
import numpy as np
import face_recognition
from bs4 import BeautifulSoup
import tldextract
from fake_useragent import UserAgent


# ================== CONFIGURATION ==================

class CrawlerConfig:
    """Configuration for web crawling."""
    MAX_PAGES_PER_USERNAME = 50
    MAX_DEPTH = 1
    TIMEOUT = 15
    MAX_WORKERS = 10
    DELAY = (1.0, 3.0)
    USER_AGENT_ROTATION = True
    FOLLOW_SAME_DOMAIN = False
    EXCLUDE_EXTENSIONS = {'.pdf', '.zip', '.tar', '.gz', '.exe', '.dmg', '.iso'}
    VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'}
    MAX_IMAGE_SIZE_MB = 5
    MAX_RETRIES = 2
    RATE_LIMIT_DELAY = 1.0
    VERBOSE = True


# ================== ENHANCED PROFILE TEMPLATES ==================

PROFILE_TEMPLATES = {
    # Tech/Developer Communities
    "github": {
        "url": "https://github.com/{}",
        "check_method": "github_check",
        "avatar_selector": "img.avatar",
        "requires_javascript": False,
        "platform": "github"
    },
    "stackoverflow": {
        "url": "https://stackoverflow.com/users/{}",
        "check_method": "stackoverflow_check",
        "avatar_selector": ".avatar img",
        "requires_javascript": False,
        "platform": "stackoverflow"
    },
    "gitlab": {
        "url": "https://gitlab.com/{}",
        "check_method": "gitlab_check",
        "avatar_selector": ".user-avatar img",
        "requires_javascript": False,
        "platform": "gitlab"
    },
    
    # Social Media (easier to scrape)
    "twitter": {
        "url": "https://twitter.com/{}",
        "check_method": "twitter_check",
        "avatar_selector": "img.css-9pa8cd",
        "requires_javascript": True,  # Twitter uses heavy JS
        "platform": "twitter"
    },
    "instagram": {
        "url": "https://www.instagram.com/{}/",
        "check_method": "instagram_check",
        "avatar_selector": "img._aadp",
        "requires_javascript": True,
        "platform": "instagram"
    },
    "reddit": {
        "url": "https://www.reddit.com/user/{}",
        "check_method": "reddit_check",
        "avatar_selector": "img._3JHAesTjqoW_RWJXOGMNq2",
        "requires_javascript": False,
        "platform": "reddit"
    },
    
    # Creative/Portfolio
    "artstation": {
        "url": "https://www.artstation.com/{}",
        "check_method": "artstation_check",
        "avatar_selector": ".profile-image img",
        "requires_javascript": False,
        "platform": "artstation"
    },
    
    # Easy to check (return proper 404s)
    "aboutme": {
        "url": "https://about.me/{}",
        "check_method": "status_code",
        "avatar_selector": ".profile_photo img",
        "requires_javascript": False,
        "platform": "aboutme"
    },
    "deviantart": {
        "url": "https://www.deviantart.com/{}",
        "check_method": "deviantart_check",
        "avatar_selector": ".avatar img",
        "requires_javascript": False,
        "platform": "deviantart"
    },
    "flickr": {
        "url": "https://www.flickr.com/people/{}",
        "check_method": "flickr_check",
        "avatar_selector": ".avatar img",
        "requires_javascript": False,
        "platform": "flickr"
    },
    "500px": {
        "url": "https://500px.com/p/{}",
        "check_method": "500px_check",
        "avatar_selector": ".avatar img",
        "requires_javascript": False,
        "platform": "500px"
    },
    
    # From your list
    "1337x": {
        "url": "https://www.1337x.to/user/{}",
        "check_method": "status_code",
        "avatar_selector": "img.avatar",
        "requires_javascript": False,
        "platform": "1337x"
    },
    "7cups": {
        "url": "https://www.7cups.com/@{}",
        "check_method": "status_code",
        "avatar_selector": ".avatar img",
        "requires_javascript": False,
        "platform": "7cups"
    },
    "archive_org": {
        "url": "https://archive.org/details/@{}",
        "check_method": "status_code",
        "avatar_selector": "img.user-avatar",
        "requires_javascript": False,
        "platform": "archive_org"
    },
    "arduino": {
        "url": "https://forum.arduino.cc/u/{}/summary",
        "check_method": "status_code",
        "avatar_selector": "img.avatar",
        "requires_javascript": False,
        "platform": "arduino"
    },
    "bandcamp": {
        "url": "https://bandcamp.com/{}",
        "check_method": "bandcamp_check",
        "avatar_selector": "#profile-image img",
        "requires_javascript": False,
        "platform": "bandcamp"
    },
    "keybase": {
        "url": "https://keybase.io/{}",
        "check_method": "keybase_check",
        "avatar_selector": "img.avatar",
        "requires_javascript": False,
        "platform": "keybase"
    },
}


# ================== SITE-SPECIFIC CHECKERS ==================

class SiteCheckers:
    """Site-specific profile existence checkers."""
    
    @staticmethod
    def github_check(response: requests.Response, username: str) -> bool:
        """Check if GitHub profile exists."""
        if response.status_code != 200:
            return False
        
        html = response.text.lower()
        
        # More accurate GitHub existence check
        not_found_indicators = [
            'this is not the web page you are looking for',
            'page not found',
            'github could not find that page',
            'there isn\'t a github pages site here',
        ]
        
        # Check for absence of profile elements
        if any(indicator in html for indicator in not_found_indicators):
            return False
        
        # Positive indicators
        profile_indicators = [
            f'itemprop="name"',
            'vcard-names-container',
            'js-profile-editable-area',
            'p-nickname vcard-username',  # GitHub username element
            'user-profile-frame',  # GitHub profile frame
        ]
        
        # Check for username in page
        if username.lower() in html:
            for indicator in profile_indicators:
                if indicator in html:
                    return True
        
        # Alternative: check for common GitHub profile elements
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check for profile-specific elements
        if soup.find('div', {'class': 'user-profile-frame'}):
            return True
        
        if soup.find('span', {'itemprop': 'name'}):
            return True
        
        # Check for the username in the page title
        title = soup.find('title')
        if title and username.lower() in title.text.lower():
            return True
        
        # Check for avatar image (GitHub avatars have specific URLs)
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if 'avatars.githubusercontent.com' in src and 'identicon' not in src:
                return True
        
        return False
    
    @staticmethod
    def stackoverflow_check(response: requests.Response, username: str) -> bool:
        """Check if Stack Overflow profile exists."""
        if response.status_code != 200:
            return False
        
        html = response.text.lower()
        
        # Stack Overflow shows "Page Not Found" for non-existent users
        if 'page not found' in html or '404 - page not found' in html:
            return False
        
        # Check for user profile elements
        profile_indicators = [
            'user-card',
            'user-avatar',
            'user-details',
        ]
        
        for indicator in profile_indicators:
            if indicator in html:
                return True
        
        return False
    
    @staticmethod
    def twitter_check(response: requests.Response, username: str) -> bool:
        """Check if Twitter profile exists."""
        # Twitter often redirects or shows different pages
        final_url = response.url.lower()
        
        # Check if we got redirected to twitter.com/home (logged out view)
        if 'twitter.com/home' in final_url:
            return False
        
        # Check if we're on the user's profile
        if f'twitter.com/{username.lower()}' in final_url:
            return True
        
        # Check response content
        html = response.text.lower()
        
        # Twitter shows "This account doesn't exist" for non-existent users
        if 'this account doesn\'t exist' in html or 'account suspended' in html:
            return False
        
        # Check for profile elements
        if 'profile-header' in html or 'user-actions' in html:
            return True
        
        return response.status_code == 200
    
    @staticmethod
    def instagram_check(response: requests.Response, username: str) -> bool:
        """Check if Instagram profile exists."""
        if response.status_code != 200:
            return False
        
        html = response.text.lower()
        
        # Instagram shows "Sorry, this page isn't available."
        if 'sorry, this page isn\'t available' in html:
            return False
        
        # Check for profile elements
        if 'profile-page' in html or 'vcard' in html:
            return True
        
        return True
    
    @staticmethod
    def reddit_check(response: requests.Response, username: str) -> bool:
        """Check if Reddit profile exists."""
        if response.status_code != 200:
            return False
        
        html = response.text.lower()
        
        # Reddit shows "page not found" or "this user has deleted their account"
        if 'page not found' in html or 'this user has deleted' in html:
            return False
        
        # Check for user profile elements
        if 'user-profile' in html or f'user/{username.lower()}' in html:
            return True
        
        return True
    
    @staticmethod
    def artstation_check(response: requests.Response, username: str) -> bool:
        """Check if ArtStation profile exists."""
        if response.status_code != 200:
            return False
        
        html = response.text.lower()
        
        # ArtStation shows "The page you were looking for doesn't exist"
        if 'doesn\'t exist' in html or 'page not found' in html:
            return False
        
        # Check for profile elements
        if 'artist-header' in html or 'user-profile' in html:
            return True
        
        return True
    
    @staticmethod
    def deviantart_check(response: requests.Response, username: str) -> bool:
        """Check if DeviantArt profile exists."""
        if response.status_code != 200:
            return False
        
        html = response.text.lower()
        
        # DeviantArt shows "The deviation you are looking for appears to be missing"
        if 'deviation you are looking for' in html or 'does not exist' in html:
            return False
        
        return True
    
    @staticmethod
    def flickr_check(response: requests.Response, username: str) -> bool:
        """Check if Flickr profile exists."""
        if response.status_code != 200:
            return False
        
        html = response.text.lower()
        
        # Flickr shows "This member is no longer active on Flickr"
        if 'no longer active' in html or 'does not exist' in html:
            return False
        
        return True
    
    @staticmethod
    def _500px_check(response: requests.Response, username: str) -> bool:
        """Check if 500px profile exists."""
        if response.status_code != 200:
            return False
        
        html = response.text.lower()
        
        # 500px shows "The page you requested could not be found"
        if 'could not be found' in html:
            return False
        
        return True
    
    @staticmethod
    def bandcamp_check(response: requests.Response, username: str) -> bool:
        """Check if Bandcamp profile exists."""
        if response.status_code != 200:
            return False
        
        html = response.text.lower()
        
        # Bandcamp shows "Couldn't find that one"
        if 'couldn\'t find that one' in html:
            return False
        
        return True
    
    @staticmethod
    def keybase_check(response: requests.Response, username: str) -> bool:
        """Check if Keybase profile exists."""
        if response.status_code != 200:
            return False
        
        html = response.text.lower()
        
        # Keybase shows "User not found"
        if 'user not found' in html:
            return False
        
        return True
    
    @staticmethod
    def gitlab_check(response: requests.Response, username: str) -> bool:
        """Check if GitLab profile exists."""
        if response.status_code == 404:
            return False
        
        if response.status_code != 200:
            return True  # GitLab might redirect or show other pages
        
        html = response.text.lower()
        
        # GitLab shows "The page could not be found" for 404s
        if 'page could not be found' in html:
            return False
        
        return True


# ================== ENHANCED PROFILE CRAWLER ==================

class EnhancedProfileCrawler:
    """Enhanced crawler with site-specific checks."""
    
    def __init__(self, config: CrawlerConfig = None):
        self.config = config or CrawlerConfig()
        self.ua = UserAgent() if self.config.USER_AGENT_ROTATION else None
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache',
            'DNT': '1',
        })
        self.checkers = SiteCheckers()
        self.rate_limit_cache = {}
    
    def get_random_user_agent(self) -> str:
        """Get a random user agent."""
        if self.ua:
            return self.ua.random
        return 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    
    def check_rate_limit(self, domain: str):
        """Rate limiting by domain."""
        current_time = time.time()
        if domain in self.rate_limit_cache:
            last_request = self.rate_limit_cache[domain]
            elapsed = current_time - last_request
            if elapsed < self.config.RATE_LIMIT_DELAY:
                sleep_time = self.config.RATE_LIMIT_DELAY - elapsed
                if self.config.VERBOSE:
                    print(f"  ⏳ Rate limiting: waiting {sleep_time:.1f}s for {domain}")
                time.sleep(sleep_time)
        
        self.rate_limit_cache[domain] = current_time
    
    def is_valid_avatar(self, url: str, img_element) -> bool:
        """Check if an image URL is likely a valid avatar (not default/placeholder)."""
        # Skip known placeholder avatars
        placeholder_keywords = [
            'default', 'placeholder', 'anonymous', 'unknown', 
            'ghost', 'blank', 'null', 'empty'
        ]
        
        url_lower = url.lower()
        for keyword in placeholder_keywords:
            if keyword in url_lower:
                return False
        
        # Check img attributes
        alt_text = (img_element.get('alt') or '').lower()
        for keyword in placeholder_keywords:
            if keyword in alt_text:
                return False
        
        # For GitHub specifically
        if 'github' in url_lower:
            # GitHub default avatars often have specific patterns
            if 'identicon' in url_lower or 'monsterid' in url_lower or 'retro' in url_lower:
                return False
        
        return True
    
    def check_profile(self, url: str, platform: str, username: str) -> Dict[str, Any]:
        """Check if a profile exists with site-specific logic."""
        domain = urlparse(url).netloc
        self.check_rate_limit(domain)
        
        # Random delay to avoid detection
        time.sleep(random.uniform(*self.config.DELAY))
        
        try:
            # Update headers for this request
            headers = {'User-Agent': self.get_random_user_agent()}
            
            response = self.session.get(
                url,
                headers=headers,
                timeout=self.config.TIMEOUT,
                allow_redirects=True,
                stream=False
            )
            
            # Get platform configuration
            platform_config = PROFILE_TEMPLATES.get(platform, {})
            if isinstance(platform_config, str):
                platform_config = {"url": platform_config, "check_method": "status_code"}
            
            check_method = platform_config.get("check_method", "status_code")
            exists = False
            
            # Use appropriate check method
            if check_method == "status_code":
                exists = response.status_code == 200
            elif check_method == "github_check":
                exists = self.checkers.github_check(response, username)
            elif check_method == "twitter_check":
                exists = self.checkers.twitter_check(response, username)
            elif check_method == "instagram_check":
                exists = self.checkers.instagram_check(response, username)
            elif check_method == "reddit_check":
                exists = self.checkers.reddit_check(response, username)
            elif check_method == "stackoverflow_check":
                exists = self.checkers.stackoverflow_check(response, username)
            elif check_method == "artstation_check":
                exists = self.checkers.artstation_check(response, username)
            elif check_method == "deviantart_check":
                exists = self.checkers.deviantart_check(response, username)
            elif check_method == "flickr_check":
                exists = self.checkers.flickr_check(response, username)
            elif check_method == "500px_check":
                exists = self.checkers._500px_check(response, username)
            elif check_method == "bandcamp_check":
                exists = self.checkers.bandcamp_check(response, username)
            elif check_method == "keybase_check":
                exists = self.checkers.keybase_check(response, username)
            elif check_method == "gitlab_check":
                exists = self.checkers.gitlab_check(response, username)
            else:
                # Default: status code 200
                exists = response.status_code == 200
            
            # Extract images if profile exists
            image_urls = []
            if exists:
                image_urls = self.extract_images(response.text, url, platform_config)
            
            result = {
                "exists": exists,
                "status_code": response.status_code,
                "url": response.url,  # Use final URL after redirects
                "image_urls": image_urls,
                "error": None,
                "platform": platform,
                "username": username,
                "final_url": response.url,
                "content_length": len(response.text)
            }
            
            return result
            
        except requests.exceptions.Timeout:
            return {
                "exists": False,
                "status_code": 408,
                "url": url,
                "image_urls": [],
                "error": "Timeout",
                "platform": platform,
                "username": username
            }
        except requests.exceptions.ConnectionError:
            return {
                "exists": False,
                "status_code": 0,
                "url": url,
                "image_urls": [],
                "error": "Connection error",
                "platform": platform,
                "username": username
            }
        except Exception as e:
            return {
                "exists": False,
                "status_code": 0,
                "url": url,
                "image_urls": [],
                "error": str(e),
                "platform": platform,
                "username": username
            }
    
    def extract_images(self, html: str, base_url: str, platform_config: Dict) -> List[str]:
        """Extract images from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        image_urls = set()
        
        # Try platform-specific selector first
        avatar_selector = platform_config.get("avatar_selector", "")
        if avatar_selector:
            try:
                for img in soup.select(avatar_selector):
                    src = self.get_image_src(img)
                    if src:
                        full_url = urljoin(base_url, src)
                        # Filter out default/placeholder avatars
                        if self.is_valid_avatar(full_url, img):
                            image_urls.add(full_url)
            except Exception as e:
                if self.config.VERBOSE:
                    print(f"    [!] Error with selector {avatar_selector}: {e}")
        
        # Special handling for GitHub
        platform_name = platform_config.get("platform", "")
        if platform_name == "github" or "github.com" in base_url:
            # Look for GitHub-specific avatar patterns
            for img in soup.find_all('img'):
                src = self.get_image_src(img)
                if not src:
                    continue
                
                # Check if it's a GitHub avatar
                if 'avatars.githubusercontent.com' in src:
                    full_url = urljoin(base_url, src)
                    # Skip default GitHub avatars
                    if 'identicon' not in src and 'monsterid' not in src and 'retro' not in src:
                        image_urls.add(full_url)
                
                # Also check for GitHub's avatar classes
                img_class = ' '.join(img.get('class', []))
                if 'avatar' in img_class.lower() and 'user' in img_class.lower():
                    full_url = urljoin(base_url, src)
                    if self.is_valid_avatar(full_url, img):
                        image_urls.add(full_url)
        
        # Also check meta tags for social images
        meta_selectors = [
            'meta[property="og:image"]',
            'meta[name="twitter:image"]',
            'meta[property="twitter:image"]',
            'meta[itemprop="image"]',
        ]
        
        for selector in meta_selectors:
            for meta in soup.select(selector):
                content = meta.get('content')
                if content:
                    try:
                        full_url = urljoin(base_url, content)
                        # Check if it looks like a profile image
                        if any(keyword in full_url.lower() for keyword in ['profile', 'avatar', 'user', 'photo']):
                            image_urls.add(full_url)
                    except:
                        pass
        
        # Fallback: all images that look like avatars
        if not image_urls:
            for img in soup.find_all('img'):
                src = self.get_image_src(img)
                if not src:
                    continue
                
                # Check if it looks like an avatar/profile image
                img_alt = (img.get('alt') or '').lower()
                img_class = (img.get('class') or [])
                img_id = (img.get('id') or '').lower()
                
                is_profile_image = any([
                    'avatar' in img_alt,
                    'profile' in img_alt,
                    'user' in img_alt,
                    any('avatar' in str(c).lower() for c in img_class),
                    any('profile' in str(c).lower() for c in img_class),
                    'avatar' in img_id,
                    'profile' in img_id,
                    'photo' in img_alt,
                    'picture' in img_alt,
                ])
                
                if is_profile_image:
                    try:
                        full_url = urljoin(base_url, src)
                        if self.is_valid_avatar(full_url, img):
                            image_urls.add(full_url)
                    except:
                        pass
        
        # Convert to list and filter
        filtered_urls = []
        for url in image_urls:
            # Skip data URIs and javascript
            if url.startswith(('data:', 'javascript:')):
                continue
            
            # Remove query parameters that might cause issues
            parsed = urlparse(url)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            
            # Check if it's likely an image
            path_lower = parsed.path.lower()
            has_image_ext = any(path_lower.endswith(ext) for ext in self.config.VALID_IMAGE_EXTENSIONS)
            
            # Also accept URLs without extensions if they're from known avatar hosts
            is_known_avatar_host = any(
                host in clean_url.lower() 
                for host in ['avatars.githubusercontent.com', 'gravatar.com', 'avatar.trakt.tv']
            )
            
            if has_image_ext or is_known_avatar_host:
                filtered_urls.append(clean_url)
        
        return list(set(filtered_urls))[:5]  # Return up to 5 unique images
    
    def get_image_src(self, img_element) -> Optional[str]:
        """Get image source from img element."""
        # Try various attributes in order
        for attr in ['src', 'data-src', 'data-original', 'srcset']:
            src = img_element.get(attr)
            if not src:
                continue
            
            if attr == 'srcset':
                # Take the first URL from srcset
                parts = src.split(',')
                if parts:
                    first_part = parts[0].strip()
                    url = first_part.split(' ')[0]
                    return url if url else None
            else:
                return src
        
        return None
    
    def crawl_usernames(self, usernames: List[str], platforms: List[str] = None) -> Dict[str, List[Dict]]:
        """Crawl multiple usernames across platforms."""
        if platforms is None:
            platforms = list(PROFILE_TEMPLATES.keys())
        
        results = {username: [] for username in usernames}
        
        with ThreadPoolExecutor(max_workers=min(self.config.MAX_WORKERS, len(platforms))) as executor:
            futures = []
            
            for username in usernames:
                username = username.strip()
                if not username:
                    continue
                
                for platform in platforms:
                    if platform not in PROFILE_TEMPLATES:
                        continue
                    
                    platform_config = PROFILE_TEMPLATES[platform]
                    if isinstance(platform_config, str):
                        url = platform_config.format(username)
                    else:
                        url = platform_config.get("url", "").format(username)
                    
                    if not url:
                        continue
                    
                    future = executor.submit(
                        self.check_profile,
                        url, platform, username
                    )
                    futures.append((future, username, platform, url))
            
            # Process results
            completed = 0
            total = len(futures)
            
            for future, username, platform, url in futures:
                try:
                    result = future.result(timeout=self.config.TIMEOUT + 10)
                    results[username].append(result)
                    completed += 1
                    
                    if self.config.VERBOSE:
                        status = "✅" if result["exists"] else "❌"
                        images = f" ({len(result['image_urls'])} img)" if result["image_urls"] else ""
                        error = f" - {result['error']}" if result["error"] else ""
                        print(f"  {status} [{completed}/{total}] {platform}: {result['exists']}{images}{error}")
                        
                except Exception as e:
                    if self.config.VERBOSE:
                        print(f"  ❌ {platform}: Error - {e}")
        
        return results


# ================== IMAGE PROCESSING ==================

def get_image_bytes(source: str, max_size_mb: int = 5, timeout: int = 10) -> Optional[bytes]:
    """Download image with error handling."""
    try:
        if source.startswith("data:"):
            b64_data = source.split(",", 1)[1]
            return base64.b64decode(b64_data)
        elif source.startswith("http://") or source.startswith("https://"):
            headers = {
                'User-Agent': UserAgent().random,
                'Accept': 'image/*,*/*;q=0.8',
            }
            
            response = requests.get(
                source, 
                headers=headers, 
                timeout=timeout, 
                stream=True,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('Content-Type', '').lower()
            if content_type and not any(x in content_type for x in ['image/', 'octet-stream', 'binary']):
                return None
            
            # Read in chunks
            content = b''
            max_bytes = max_size_mb * 1024 * 1024
            
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > max_bytes:
                    return None
            
            return content
        else:
            if not os.path.exists(source):
                return None
            
            with open(source, "rb") as f:
                return f.read()
    except Exception:
        return None


def compute_face_encoding(image_bytes: bytes) -> Optional[np.ndarray]:
    """Extract face encoding from image."""
    try:
        image = Image.open(BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        rgb_image = np.array(image)
        
        # Try face detection
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        if not face_locations:
            return None
        
        encodings = face_recognition.face_encodings(rgb_image, face_locations)
        return encodings[0] if encodings else None
        
    except Exception:
        return None


# ================== FACE INDEX SYSTEM ==================

class FaceIndexSystem:
    """Face indexing system."""
    
    def __init__(self):
        self.faces = []
        self.config = CrawlerConfig()
    
    def index_from_results(self, crawl_results: Dict[str, List[Dict]]) -> List[Dict]:
        """Index faces from crawl results."""
        new_faces = []
        
        for username, results in crawl_results.items():
            for result in results:
                if not result["exists"]:
                    continue
                
                for image_url in result["image_urls"][:2]:  # Try first 2 images
                    try:
                        img_bytes = get_image_bytes(image_url)
                        if not img_bytes:
                            continue
                        
                        encoding = compute_face_encoding(img_bytes)
                        if encoding is None:
                            continue
                        
                        face_record = {
                            "username": username,
                            "platform": result["platform"],
                            "page_url": result["url"],
                            "image_url": image_url,
                            "encoding": encoding.tolist(),
                            "timestamp": time.time()
                        }
                        
                        self.faces.append(face_record)
                        new_faces.append(face_record)
                        
                        if self.config.VERBOSE:
                            print(f"    👤 Face indexed: {username}@{result['platform']}")
                        
                    except Exception as e:
                        if self.config.VERBOSE:
                            print(f"    [!] Error: {e}")
        
        return new_faces
    
    def search_faces(self, target_encoding: np.ndarray, threshold: float = 0.6, top_k: int = 10) -> List[Dict]:
        """Search for similar faces."""
        results = []
        
        for face in self.faces:
            try:
                face_encoding = np.array(face["encoding"])
                distance = float(face_recognition.face_distance([target_encoding], face_encoding)[0])
                similarity = max(0.0, 1.0 - min(distance, 1.0))
                
                results.append({
                    "username": face["username"],
                    "platform": face["platform"],
                    "similarity": similarity,
                    "distance": distance,
                    "match": distance < threshold,
                    "page_url": face["page_url"],
                    "image_url": face["image_url"]
                })
            except Exception:
                continue
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def save_index(self, filename: str = "face_index.json"):
        """Save index to file."""
        data = {
            "faces": self.faces,
            "metadata": {
                "total": len(self.faces),
                "timestamp": time.time()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved {len(self.faces)} faces to {filename}")
    
    def load_index(self, filename: str = "face_index.json"):
        """Load index from file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.faces = data.get("faces", [])
            print(f"📂 Loaded {len(self.faces)} faces from {filename}")
            return True
        except Exception as e:
            print(f"❌ Error loading: {e}")
            return False


# ================== TESTING ==================

def test_known_profiles():
    """Test with known profiles."""
    print("🧪 Testing with known profiles...")
    
    test_cases = [
        ("torvalds", "github", True, "Linus Torvalds"),
        ("jack", "twitter", True, "Jack Dorsey"),
        ("nasdaily", "instagram", True, "Nas Daily"),
        ("spez", "reddit", True, "Reddit CEO"),
        ("beeple", "artstation", True, "Digital artist"),
        ("nonexistent1234567890", "github", False, "Non-existent"),
    ]
    
    crawler = EnhancedProfileCrawler(CrawlerConfig())
    crawler.config.VERBOSE = False
    
    passed = 0
    failed = 0
    
    for username, platform, should_exist, description in test_cases:
        if platform not in PROFILE_TEMPLATES:
            print(f"  ⚠️  Skipping {platform} (not configured)")
            continue
        
        platform_config = PROFILE_TEMPLATES[platform]
        if isinstance(platform_config, str):
            url = platform_config.format(username)
        else:
            url = platform_config.get("url", "").format(username)
        
        print(f"\n🔍 {username} on {platform} ({description}):")
        print(f"  URL: {url}")
        
        result = crawler.check_profile(url, platform, username)
        
        status = "✅ PASS" if result["exists"] == should_exist else "❌ FAIL"
        if result["exists"] == should_exist:
            passed += 1
        else:
            failed += 1
        
        print(f"  {status} - Expected: {should_exist}, Got: {result['exists']}")
        print(f"  Status: {result['status_code']}, Images: {len(result['image_urls'])}")
        
        if result["error"]:
            print(f"  Error: {result['error']}")
        
        # Show first image if exists
        if result["image_urls"]:
            print(f"  First image: {result['image_urls'][0][:80]}...")
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")


def test_specific_profile():
    """Test a specific profile."""
    username = input("Username: ").strip()
    platform = input("Platform: ").strip()
    
    if not username or not platform:
        print("❌ Need username and platform")
        return
    
    if platform not in PROFILE_TEMPLATES:
        print(f"❌ Unknown platform. Available: {', '.join(list(PROFILE_TEMPLATES.keys())[:10])}...")
        return
    
    crawler = EnhancedProfileCrawler()
    
    platform_config = PROFILE_TEMPLATES[platform]
    if isinstance(platform_config, str):
        url = platform_config.format(username)
    else:
        url = platform_config.get("url", "").format(username)
    
    print(f"\n🔍 Testing {username} on {platform}...")
    print(f"  URL: {url}")
    
    result = crawler.check_profile(url, platform, username)
    
    print(f"\n📊 Results:")
    print(f"  Exists: {result['exists']}")
    print(f"  Status Code: {result['status_code']}")
    print(f"  Final URL: {result['final_url']}")
    print(f"  Content Length: {result['content_length']} chars")
    print(f"  Images Found: {len(result['image_urls'])}")
    
    if result["error"]:
        print(f"  Error: {result['error']}")
    
    # Show images
    for i, img_url in enumerate(result["image_urls"][:3], 1):
        print(f"\n  Image {i}:")
        print(f"    URL: {img_url}")
        
        # Try to download and check
        img_bytes = get_image_bytes(img_url)
        if img_bytes:
            print(f"    Size: {len(img_bytes)} bytes")
            encoding = compute_face_encoding(img_bytes)
            if encoding is not None:
                print(f"    ✅ Face detected")
            else:
                print(f"    ❌ No face detected")
        else:
            print(f"    ❌ Could not download")


# ================== MAIN INTERFACE ==================

def main():
    """Main interface."""
    print("🔍 Enhanced Cross-Platform Face Search")
    print("=" * 60)
    
    # Initialize
    crawler = EnhancedProfileCrawler()
    face_system = FaceIndexSystem()
    
    # Load existing index
    if os.path.exists("face_index.json"):
        face_system.load_index()
    
    while True:
        print("\n" + "=" * 60)
        print("1. Search for usernames")
        print("2. Test specific profile")
        print("3. Run known profile tests")
        print("4. Compare target face")
        print("5. Show statistics")
        print("6. Save index")
        print("7. Load index")
        print("8. Clear index")
        print("9. Exit")
        
        choice = input("\nSelect option (1-9): ").strip()
        
        if choice == "1":
            # Search usernames
            usernames_input = input("Enter usernames (comma-separated): ").strip()
            if not usernames_input:
                continue
            
            usernames = [u.strip() for u in usernames_input.split(',')]
            
            # Platform selection
            platforms = list(PROFILE_TEMPLATES.keys())
            print(f"\nAvailable platforms ({len(platforms)}):")
            print(", ".join(platforms))
            
            platform_input = input("\nEnter platforms (comma-separated, or 'all'): ").strip().lower()
            
            if platform_input == 'all':
                selected_platforms = platforms
            else:
                selected_platforms = []
                for item in platform_input.split(','):
                    item = item.strip()
                    if item in platforms:
                        selected_platforms.append(item)
            
            if not selected_platforms:
                # Default to platforms that work well
                selected_platforms = ["github", "twitter", "reddit", "aboutme", "artstation", "deviantart"]
            
            print(f"\n🔍 Searching {len(usernames)} user(s) on {len(selected_platforms)} platform(s)...")
            
            # Crawl
            results = crawler.crawl_usernames(usernames, selected_platforms)
            
            # Index faces
            print("\n📸 Indexing faces...")
            new_faces = face_system.index_from_results(results)
            
            # Summary
            print(f"\n📊 Summary:")
            total_found = 0
            total_faces = 0
            
            for username in usernames:
                user_results = results.get(username, [])
                found = [r for r in user_results if r["exists"]]
                user_faces = len([f for f in new_faces if f["username"] == username])
                
                total_found += len(found)
                total_faces += user_faces
                
                print(f"  {username}: {len(found)}/{len(user_results)} profiles, {user_faces} faces")
            
            print(f"\n  Total: {total_found} profiles found, {total_faces} faces indexed")
            
            # Offer to save
            if new_faces:
                save = input("\n💾 Save results to index? (y/N): ").strip().lower()
                if save == 'y':
                    face_system.save_index()
        
        elif choice == "2":
            test_specific_profile()
        
        elif choice == "3":
            test_known_profiles()
        
        elif choice == "4":
            if not face_system.faces:
                print("❌ No faces in index")
                continue
            
            target_source = input("Target image path or URL: ").strip()
            if not target_source:
                continue
            
            # Load target image
            target_bytes = get_image_bytes(target_source)
            if not target_bytes:
                print("❌ Could not load image")
                continue
            
            target_encoding = compute_face_encoding(target_bytes)
            if target_encoding is None:
                print("❌ No face detected in target")
                continue
            
            # Get threshold
            try:
                threshold = float(input("Match threshold (0.1-1.0, default 0.6): ") or "0.6")
                top_k = int(input("Number of results (default 10): ") or "10")
            except ValueError:
                threshold = 0.6
                top_k = 10
            
            print(f"\n🔍 Searching {len(face_system.faces)} indexed faces...")
            matches = face_system.search_faces(target_encoding, threshold, top_k)
            
            if not matches:
                print("❌ No matches found")
                continue
            
            print(f"\n🏆 Top {len(matches)} matches:")
            for i, match in enumerate(matches, 1):
                symbol = "✅" if match["match"] else "⚠️"
                print(f"\n  {i}. {symbol} Similarity: {match['similarity']:.3f}")
                print(f"     User: {match['username']}")
                print(f"     Platform: {match['platform']}")
                if match['similarity'] > 0.7:
                    print(f"     🎯 Strong match!")
        
        elif choice == "5":
            print(f"\n📊 Statistics:")
            print(f"  Total faces: {len(face_system.faces)}")
            
            if face_system.faces:
                # Count by platform
                platforms = {}
                for face in face_system.faces:
                    platform = face.get("platform", "unknown")
                    platforms[platform] = platforms.get(platform, 0) + 1
                
                print(f"  By platform:")
                for platform, count in sorted(platforms.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {platform}: {count}")
        
        elif choice == "6":
            filename = input("Filename (default: face_index.json): ").strip() or "face_index.json"
            face_system.save_index(filename)
        
        elif choice == "7":
            filename = input("Filename (default: face_index.json): ").strip() or "face_index.json"
            face_system.load_index(filename)
        
        elif choice == "8":
            confirm = input("Clear all indexed faces? (y/N): ").strip().lower()
            if confirm == 'y':
                face_system.faces = []
                print("✅ Index cleared")
        
        elif choice == "9":
            print("👋 Goodbye!")
            break


if __name__ == "__main__":
    main()