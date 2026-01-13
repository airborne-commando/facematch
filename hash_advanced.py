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
    PROFILE_TEMPLATES_FILE = "profile_templates.json"


# ================== LOAD PROFILE TEMPLATES FROM JSON ==================

def load_profile_templates(filename: str = "profile_templates.json") -> Dict[str, Any]:
    """
    Load profile templates from a JSON file.
    If file doesn't exist, return empty dict.
    """
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                templates = json.load(f)
                print(f"✅ Loaded {len(templates)} profile templates from {filename}")
                return templates
        else:
            print(f"❌ Profile templates file not found: {filename}")
            print("   Create a JSON file with platform configurations.")
            return {}
    except Exception as e:
        print(f"❌ Error loading profile templates from {filename}: {e}")
        return {}


def save_profile_templates(templates: Dict[str, Any], filename: str = "profile_templates.json"):
    """Save profile templates to JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(templates, f, indent=2)
        print(f"💾 Saved {len(templates)} profile templates to {filename}")
    except Exception as e:
        print(f"❌ Error saving profile templates: {e}")


def get_enabled_platforms(templates: Dict[str, Any]) -> List[str]:
    """Get list of enabled platforms from templates."""
    enabled = []
    for platform_name, config in templates.items():
        if config.get("enabled", True):
            enabled.append(platform_name)
    return enabled


def get_platforms_by_category(templates: Dict[str, Any]) -> Dict[str, List[str]]:
    """Organize platforms by category."""
    categories = {}
    for platform_name, config in templates.items():
        if config.get("enabled", True):
            category = config.get("category", "Uncategorized")
            if category not in categories:
                categories[category] = []
            categories[category].append(platform_name)
    return categories


# Load templates at module level
PROFILE_TEMPLATES = load_profile_templates()


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
            'p-nickname vcard-username',
            'user-profile-frame',
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
    
    @staticmethod
    def universal_check(response: requests.Response, username: str) -> bool:
        """Universal profile checker for any site."""
        if response.status_code != 200:
            return False
        
        html = response.text.lower()
        
        # Common "not found" patterns across many sites
        not_found_indicators = [
            'page not found',
            '404',
            'not found',
            'doesn\'t exist',
            'does not exist',
            'couldn\'t be found',
            'no longer available',
            'user not found',
            'profile not found',
            'account not found',
            'this page could not be found',
            'sorry, this page isn\'t available',
            'the page you were looking for',
            'we couldn\'t find that page',
        ]
        
        for indicator in not_found_indicators:
            if indicator in html:
                return False
        
        # Check for username in page (good indicator of profile page)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check title
        title = soup.find('title')
        if title and username.lower() in title.text.lower():
            return True
        
        # Check meta tags
        for meta in soup.find_all('meta'):
            content = meta.get('content', '').lower()
            if username.lower() in content:
                return True
        
        # Check for common profile elements
        profile_keywords = ['profile', 'user', 'member', 'account', 'avatar']
        page_text = soup.get_text().lower()
        
        for keyword in profile_keywords:
            if keyword in page_text:
                # Also check if username appears near profile keyword
                text_lower = page_text.replace('\n', ' ').replace('\r', ' ')
                if username.lower() in text_lower:
                    return True
        
        # Default to True if we got a 200 and no "not found" indicators
        return True

    @staticmethod
    def fansfinder_check(response: requests.Response, username: str) -> bool:
        """Check if OnlyFans profile exists via FansFinder."""
        html = response.text.lower()
        
        # Check for specific indicators that the profile exists on FansFinder
        existence_indicators = [
            f'data-username="{username.lower()}"',  # Username in data attribute
            f'onlyfans.com/{username.lower()}',  # Profile URL in page
            "media.onlyfinder.com",  # OnlyFans content URLs via FansFinder
            "og:title",  # Open Graph tags
            "og:description",  # Open Graph tags
            "user-profile profile-container",  # Profile container
            "avatar-container",  # Avatar container
            "about-profile",  # User about section
            "profile-icon",  # Profile icons
            "img-responsive",  # Responsive images
        ]
        
        # Check for "profile not found" indicators
        not_found_indicators = [
            "page not found",
            "profile not found",
            "doesn't exist",
            "does not exist",
            "no longer active",
            "user not found",
            "couldn't find that profile",
            "this profile is not available",
            "no results found",
            "no profiles found",
            "0 results",
        ]
        
        # If we see "not found" indicators, profile doesn't exist
        for indicator in not_found_indicators:
            if indicator in html:
                return False
        
        # Check for existence indicators
        for indicator in existence_indicators:
            if indicator in html:
                return True
        
        # Also check for specific patterns in the HTML structure
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check for the specific FansFinder profile container
        profile_containers = soup.find_all('div', {'class': re.compile(r'user-profile.*profile-container')})
        if profile_containers:
            for container in profile_containers:
                # Check if username is in container's data attributes
                data_username = container.get('data-username', '')
                if username.lower() == data_username.lower():
                    return True
                
                # Check for OnlyFans link in the container
                onlyfans_links = container.find_all('a', href=True)
                for link in onlyfans_links:
                    if f'onlyfans.com/{username.lower()}' in link.get('href', '').lower():
                        return True
        
        # Check for avatar images
        avatar_images = soup.find_all('img', {'class': 'img-responsive'})
        if avatar_images:
            # Check if any image has the username in alt or title
            for img in avatar_images:
                alt_text = img.get('alt', '').lower()
                title_text = img.get('title', '').lower()
                if username.lower() in alt_text or username.lower() in title_text:
                    return True
        
        # Check for profile headers with the username
        profile_headers = soup.find_all(['h1', 'h2', 'h3', 'h4'])
        for header in profile_headers:
            if username.lower() in header.get_text().lower():
                return True
        
        # Check response status
        if response.status_code == 200:
            # Additional check: look for FansFinder specific elements
            if 'fansfinder' in response.url.lower():
                # Check if we have meaningful content (not just a search page)
                if len(response.text) > 5000:  # Profile pages tend to be larger
                    # Look for profile-specific data
                    if 'profile-container' in html or 'avatar-container' in html:
                        return True
        
        return False


# ================== ENHANCED PROFILE CRAWLER ==================

class EnhancedProfileCrawler:
    """Enhanced crawler with site-specific checks and universal image extraction."""
    
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
        self.profile_templates = load_profile_templates(self.config.PROFILE_TEMPLATES_FILE)
    
    def get_random_user_agent(self) -> str:
        """Get a random user agent."""
        if self.ua:
            return self.ua.random
        return 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    
    def get_browser_like_headers(self) -> Dict[str, str]:
        """Get headers that look like a real browser."""
        return {
            'User-Agent': self.get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'TE': 'trailers',
        }
    
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
        """Universal check if an image is likely a valid avatar."""
        url_lower = url.lower()
        
        # Skip known placeholder/blank avatars
        placeholder_keywords = [
            'default', 'placeholder', 'anonymous', 'unknown', 
            'ghost', 'blank', 'null', 'empty', 'none',
            'no-avatar', 'no-avatar.jpg', 'no-photo', 'no-image',
            'default_avatar', 'default-profile', 'default-user',
            'gravatar.com/avatar/?',  # Empty gravatar
            'identicon', 'monsterid', 'wavatar', 'retro',  # GitHub defaults
            '0.jpg', '0.png', '0.gif',  # Zero filenames
        ]
        
        for keyword in placeholder_keywords:
            if keyword in url_lower:
                return False
        
        # Check image element attributes
        alt_text = (img_element.get('alt') or '').lower()
        for keyword in placeholder_keywords:
            if keyword in alt_text:
                return False
        
        title_text = (img_element.get('title') or '').lower()
        for keyword in placeholder_keywords:
            if keyword in title_text:
                return False
        
        # Check for common placeholder dimensions (very small images)
        try:
            width_attr = img_element.get('width', '')
            height_attr = img_element.get('height', '')
            
            if width_attr and height_attr:
                width = int(''.join(filter(str.isdigit, width_attr)) or '0')
                height = int(''.join(filter(str.isdigit, height_attr)) or '0')
                
                # Skip very small images (likely icons, not avatars)
                if width < 32 or height < 32:
                    return False
        except:
            pass
        
        # Check for common placeholder class names
        img_class = ' '.join(img_element.get('class', [])).lower()
        placeholder_classes = [
            'placeholder', 'default', 'empty', 'blank',
            'no-avatar', 'no-image', 'avatar-placeholder'
        ]
        
        for p_class in placeholder_classes:
            if p_class in img_class:
                return False
        
        # Platform-specific checks
        # GitHub
        if 'github' in url_lower:
            if any(x in url_lower for x in ['identicon', 'monsterid', 'retro', 'wavatar']):
                return False
        
        # Gravatar
        if 'gravatar.com/avatar/' in url_lower:
            # Check for MD5 hash length (32 chars) - empty gravatars have short or no hash
            import re
            match = re.search(r'gravatar\.com/avatar/([a-fA-F0-9]+)', url_lower)
            if match:
                hash_value = match.group(1)
                if len(hash_value) < 32:  # Not a proper MD5 hash
                    return False
        
        return True
    
    def get_image_src(self, img_element) -> Optional[str]:
        """Get image source from img element with priority order."""
        # List of attributes to check in order of priority
        src_attrs = ['src', 'data-src', 'data-original', 'data-lazy-src', 'data-lazyload', 'srcset']
        
        for attr in src_attrs:
            src = img_element.get(attr)
            if not src:
                continue
            
            # Clean and return the URL
            if attr == 'srcset':
                # Handle srcset - take the first (usually highest quality) image
                srcset_parts = src.split(',')
                if srcset_parts:
                    first_part = srcset_parts[0].strip()
                    url = first_part.split(' ')[0].strip()
                    return url if url else None
            else:
                # For other attributes, return directly
                return src.strip()
        
        return None
    
    def extract_fansfinder_avatar(self, html: str, base_url: str, username: str) -> List[str]:
        """Extract avatar from FansFinder profile page for specific username."""
        soup = BeautifulSoup(html, 'html.parser')
        image_urls = set()
        
        # Look for the specific avatar container structure
        avatar_containers = soup.find_all('div', {'class': 'avatar-container'})
        
        for container in avatar_containers:
            # Check if this container belongs to our username
            # Look for data-username attribute in parent containers
            parent = container.find_parent('div', {'data-username': username.lower()})
            if parent:
                # This container belongs to our username
                images = container.find_all('img')
                for img in images:
                    src = self.get_image_src(img)
                    if src:
                        try:
                            full_url = urljoin(base_url, src)
                            if self.is_valid_avatar(full_url, img):
                                image_urls.add(full_url)
                        except Exception as e:
                            if self.config.VERBOSE:
                                print(f"    [!] URL join error: {e}")
        
        # Also look for images with username patterns in URLs and attributes
        for img in soup.find_all('img'):
            src = self.get_image_src(img)
            if not src:
                continue
            
            src_lower = src.lower()
            username_lower = username.lower()
            
            # Check if image URL contains the username
            if username_lower in src_lower:
                # Check for specific patterns
                patterns = [
                    f'{username_lower}-onlyfans.',
                    f'{username_lower}_onlyfans.',
                    f'/{username_lower}/',
                    f'/{username_lower}-',
                ]
                
                for pattern in patterns:
                    if pattern in src_lower:
                        try:
                            full_url = urljoin(base_url, src)
                            if self.is_valid_avatar(full_url, img):
                                image_urls.add(full_url)
                                break
                        except Exception as e:
                            if self.config.VERBOSE:
                                print(f"    [!] URL join error: {e}")
            
            # Check alt and title attributes
            alt = img.get('alt', '').lower()
            title = img.get('title', '').lower()
            
            # If alt or title contains the username and "onlyfans"
            if username_lower in alt and 'onlyfans' in alt:
                try:
                    full_url = urljoin(base_url, src)
                    if self.is_valid_avatar(full_url, img):
                        image_urls.add(full_url)
                except Exception as e:
                    if self.config.VERBOSE:
                        print(f"    [!] URL join error: {e}")
            
            if username_lower in title and 'onlyfans' in title:
                try:
                    full_url = urljoin(base_url, src)
                    if self.is_valid_avatar(full_url, img):
                        image_urls.add(full_url)
                except Exception as e:
                    if self.config.VERBOSE:
                        print(f"    [!] URL join error: {e}")
        
        # If we still don't have images, look for the most likely profile image
        if not image_urls:
            # Look for images with img-responsive class
            for img in soup.find_all('img', {'class': 'img-responsive'}):
                src = self.get_image_src(img)
                if src:
                    try:
                        full_url = urljoin(base_url, src)
                        if self.is_valid_avatar(full_url, img):
                            image_urls.add(full_url)
                    except Exception as e:
                        if self.config.VERBOSE:
                            print(f"    [!] URL join error: {e}")
        
        return list(image_urls)
    
    def check_profile_with_cf_bypass(self, url: str, platform: str, username: str) -> Dict[str, Any]:
        """Check profile with Cloudflare bypass attempts."""
        domain = urlparse(url).netloc
        self.check_rate_limit(domain)
        
        time.sleep(random.uniform(*self.config.DELAY))
        
        try:
            headers = self.get_browser_like_headers()
            
            response = self.session.get(
                url,
                headers=headers,
                timeout=self.config.TIMEOUT,
                allow_redirects=True,
                stream=False
            )
            
            # For OnlyFans specifically, use fansfinder_check
            exists = False
            if platform == "onlyfans":
                exists = self.checkers.fansfinder_check(response, username)
            else:
                # Use standard check for other platforms
                platform_config = self.profile_templates.get(platform, {})
                check_method = platform_config.get("check_method", "status_code")
                
                if check_method == "status_code":
                    exists = response.status_code == 200
                else:
                    # Use the appropriate checker method
                    check_method_name = f"{check_method}"
                    if hasattr(self.checkers, check_method_name):
                        checker_func = getattr(self.checkers, check_method_name)
                        exists = checker_func(response, username)
            
            # Extract images if profile exists
            image_urls = []
            if exists:
                platform_config = self.profile_templates.get(platform, {})
                if platform == "onlyfans":
                    # Use the updated method that takes username
                    image_urls = self.extract_fansfinder_avatar(response.text, url, username)
                else:
                    image_urls = self.extract_images(response.text, url, platform_config)
            
            result = {
                "exists": exists,
                "status_code": response.status_code,
                "url": response.url,
                "image_urls": image_urls,
                "error": None,
                "platform": platform,
                "username": username,
                "final_url": response.url,
                "content_length": len(response.text),
                "cf_protected": "cf-ray" in response.headers  # Indicate if Cloudflare was detected
            }
            
            return result
            
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
    
    def check_profile(self, url: str, platform: str, username: str) -> Dict[str, Any]:
        """Check if a profile exists with site-specific logic."""
        # Use special handling for OnlyFans (using FansFinder)
        if platform == "onlyfans":
            return self.check_profile_with_cf_bypass(url, platform, username)
        
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
            platform_config = self.profile_templates.get(platform, {})
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
            elif check_method == "universal_check":
                exists = self.checkers.universal_check(response, username)
            elif check_method == "fansfinder_check":
                exists = self.checkers.fansfinder_check(response, username)
            else:
                # Default: status code 200
                exists = response.status_code == 200
            
            # Extract images if profile exists
            image_urls = []
            if exists:
                image_urls = self.extract_images(response.text, url, platform_config, username)
            
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
        """Universal image extraction that works with any site."""
        soup = BeautifulSoup(html, 'html.parser')
        image_urls = set()
        
        # Get platform name for specific handling if needed
        platform_name = platform_config.get("platform", "")
        platform_url = platform_config.get("url", "")
        
        # Special handling for OnlyFans/FansFinder
        if platform_name == "onlyfans" and username:
            fansfinder_images = self.extract_fansfinder_avatar(html, base_url, username)
            image_urls.update(fansfinder_images)
        
        # Phase 1: Try platform-specific selector first
        avatar_selector = platform_config.get("avatar_selector", "")
        if avatar_selector:
            try:
                for img in soup.select(avatar_selector):
                    src = self.get_image_src(img)
                    if src:
                        try:
                            full_url = urljoin(base_url, src)
                            if self.is_valid_avatar(full_url, img):
                                image_urls.add(full_url)
                        except Exception as e:
                            if self.config.VERBOSE:
                                print(f"    [!] URL join error: {e}")
            except Exception as e:
                if self.config.VERBOSE:
                    print(f"    [!] Error with selector {avatar_selector}: {e}")
        
        # Phase 2: Universal avatar detection patterns
        if not image_urls:
            # Pattern 1: Look for common avatar class patterns
            avatar_patterns = [
                r'.*avatar.*',  # Contains 'avatar'
                r'.*profile.*',  # Contains 'profile'
                r'.*user.*',  # Contains 'user'
                r'.*photo.*',  # Contains 'photo'
                r'.*pic.*',  # Contains 'pic'
                r'.*image.*',  # Contains 'image'
            ]
            
            for img in soup.find_all('img'):
                # Check class attribute
                img_class = ' '.join(img.get('class', []))
                if any(re.search(pattern, img_class, re.IGNORECASE) for pattern in avatar_patterns):
                    src = self.get_image_src(img)
                    if src:
                        try:
                            full_url = urljoin(base_url, src)
                            if self.is_valid_avatar(full_url, img):
                                image_urls.add(full_url)
                        except:
                            pass
                
                # Check id attribute
                img_id = img.get('id', '')
                if any(re.search(pattern, img_id, re.IGNORECASE) for pattern in avatar_patterns):
                    src = self.get_image_src(img)
                    if src:
                        try:
                            full_url = urljoin(base_url, src)
                            if self.is_valid_avatar(full_url, img):
                                image_urls.add(full_url)
                        except:
                            pass
                
                # Check alt attribute
                img_alt = img.get('alt', '').lower()
                if any(keyword in img_alt for keyword in ['profile', 'avatar', 'user', 'photo', 'picture']):
                    src = self.get_image_src(img)
                    if src:
                        try:
                            full_url = urljoin(base_url, src)
                            if self.is_valid_avatar(full_url, img):
                                image_urls.add(full_url)
                        except:
                            pass
        
        # Phase 3: Look for meta tags (social sharing images)
        meta_selectors = [
            'meta[property="og:image"]',
            'meta[name="og:image"]',
            'meta[property="twitter:image"]',
            'meta[name="twitter:image"]',
            'meta[itemprop="image"]',
            'meta[name="image"]',
        ]
        
        for selector in meta_selectors:
            for meta in soup.select(selector):
                content = meta.get('content')
                if content:
                    try:
                        full_url = urljoin(base_url, content)
                        # Check if it looks like a profile image
                        if self.config.VERBOSE:
                            print(f"    📱 Found meta image: {full_url}")
                        image_urls.add(full_url)
                    except:
                        pass
        
        # Phase 4: Check all images with common avatar filename patterns
        if not image_urls:
            avatar_filename_patterns = [
                r'.*avatar.*\.(jpg|jpeg|png|gif|webp)$',
                r'.*profile.*\.(jpg|jpeg|png|gif|webp)$',
                r'.*user.*\.(jpg|jpeg|png|gif|webp)$',
                r'.*pic.*\.(jpg|jpeg|png|gif|webp)$',
                r'.*photo.*\.(jpg|jpeg|png|gif|webp)$',
                r'.*pfp.*\.(jpg|jpeg|png|gif|webp)$',  # pfp = profile picture
                r'.*me.*\.(jpg|jpeg|png|gif|webp)$',
            ]
            
            for img in soup.find_all('img'):
                src = self.get_image_src(img)
                if not src:
                    continue
                
                # Extract filename from URL
                filename = src.split('/')[-1].split('?')[0].lower()
                
                # Check if filename matches avatar patterns
                if any(re.search(pattern, filename, re.IGNORECASE) for pattern in avatar_filename_patterns):
                    try:
                        full_url = urljoin(base_url, src)
                        if self.is_valid_avatar(full_url, img):
                            image_urls.add(full_url)
                    except:
                        pass
        
        # Phase 5: Check all images with common avatar URL patterns
        if not image_urls:
            avatar_url_patterns = [
                r'.*\/avatar\/.*',
                r'.*\/profile\/.*',
                r'.*\/user\/.*',
                r'.*\/photo\/.*',
                r'gravatar\.com\/avatar\/.*',
                r'avatars\..*\.com\/.*',
                r'cdn\.discordapp\.com\/avatars\/.*',
                r'ugc\.production\.linktr\.ee\/.*',  # Linktree CDN
                r'pbs\.twimg\.com\/profile_images\/.*',  # Twitter
                r'instagram\.fbom.*\.fna\.fbcdn\.net\/.*',  # Instagram
                r'i\.redd\.it\/.*',  # Reddit
                r'i\.imgur\.com\/.*',  # Imgur
                r'media\.onlyfinder\.com\/.*',  # FansFinder/OnlyFans CDN
            ]
            
            for img in soup.find_all('img'):
                src = self.get_image_src(img)
                if not src:
                    continue
                
                # Check if URL matches avatar patterns
                if any(re.search(pattern, src, re.IGNORECASE) for pattern in avatar_url_patterns):
                    try:
                        full_url = urljoin(base_url, src)
                        if self.is_valid_avatar(full_url, img):
                            image_urls.add(full_url)
                    except:
                        pass
        
        # Phase 6: Fallback - take first few images that look reasonable
        if not image_urls:
            for img in soup.find_all('img')[:10]:  # Limit to first 10 images
                src = self.get_image_src(img)
                if not src:
                    continue
                
                try:
                    full_url = urljoin(base_url, src)
                    
                    # Check if it's a reasonable size (not an icon)
                    img_width = img.get('width')
                    img_height = img.get('height')
                    
                    # If dimensions are specified, check if it's avatar-sized
                    if img_width and img_height:
                        width = int(img_width) if img_width.isdigit() else 0
                        height = int(img_height) if img_height.isdigit() else 0
                        
                        # Avatars are usually square-ish and not tiny
                        if width > 50 and height > 50:
                            if self.is_valid_avatar(full_url, img):
                                image_urls.add(full_url)
                    else:
                        # No dimensions, just add it
                        if self.is_valid_avatar(full_url, img):
                            image_urls.add(full_url)
                            
                except:
                    pass
        
        # Filter and clean URLs
        filtered_urls = []
        for url in image_urls:
            # Skip data URIs and javascript
            if url.startswith(('data:', 'javascript:')):
                continue
            
            # Remove query parameters that might cause issues
            try:
                parsed = urlparse(url)
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                
                # Check if it's likely an image
                path_lower = parsed.path.lower()
                has_image_ext = any(path_lower.endswith(ext) for ext in self.config.VALID_IMAGE_EXTENSIONS)
                
                # Expanded list with more mainstream and frequently used hosts
                is_known_avatar_host = any(
                    host in clean_url.lower() 
                    for host in [
                        'avatars.githubusercontent.com',      # GitHub
                        'gravatar.com',                      # Gravatar
                        'avatar.trakt.tv',                   # Trakt
                        'ugc.production.linktr.ee',          # Linktree
                        'cdn.discordapp.com',                # Discord
                        'pbs.twimg.com/profile_images',       # Twitter/X
                        'instagram.fbom1-2.fna.fbcdn.net',   # Instagram (Facebook CDN)
                        'scontent.cdninstagram.com',         # Instagram
                        'i.redd.it',                         # Reddit
                        'i.imgur.com',                       # Imgur
                        'public.onlyfans.com/files',         # OnlyFans
                        'media.onlyfinder.com',              # FansFinder/OnlyFans CDN
                        # Additional mainstream hosts
                        'platform.twitter.com',              # Twitter CDN variant
                        'abs.twimg.com',                      # Twitter avatars
                        'lh3.googleusercontent.com',          # Google/YouTube
                        'yt3.ggpht.com',                     # YouTube
                        'a0.muscdn.com',                     # SoundCloud
                        'i1.sndcdn.com',                     # SoundCloud
                        'a.pomf.lol',                        # Pomf.cat (meme culture)
                        'pbs.twimg.com/media',               # Twitter media (profile pics often here)
                        'via.placeholder.com',               # Common placeholder service
                        'ui-avatars.com',                    # Generated avatars
                        'robohash.org',                      # Robot avatars
                        'identicons.github.com',             # GitHub identicons
                        'secure.gravatar.com/avatar',        # Gravatar secure
                        'steamcdn-a.akamaihd.net',           # Steam
                        'steamuserimages-a.akamaihd.net',    # Steam
                        'avatar-management--avatars.us-west-2',  # Twitch
                        'static-cdn.jtvnw.net',              # Twitch
                        'tiktokcdn.com',                     # TikTok
                        'byteimg.com',                       # TikTok CDN
                        'ssl-profile-images-cdn.viago.co',   # LinkedIn variant
                        'media.licdn.com/dms/image',          # LinkedIn
                        'https://media.licdn.com/dms/image/v2/'
                    ]
                )
                
                if has_image_ext or is_known_avatar_host:
                    filtered_urls.append(clean_url)
            except:
                continue
        
        # Return unique URLs, limited to reasonable number
        return list(set(filtered_urls))[:10]  # Return up to 10 unique images
    
    def crawl_usernames(self, usernames: List[str], platforms: List[str] = None) -> Dict[str, List[Dict]]:
        """Crawl multiple usernames across platforms."""
        if platforms is None:
            platforms = get_enabled_platforms(self.profile_templates)
        
        results = {username: [] for username in usernames}
        
        with ThreadPoolExecutor(max_workers=min(self.config.MAX_WORKERS, len(platforms))) as executor:
            futures = []
            
            for username in usernames:
                username = username.strip()
                if not username:
                    continue
                
                for platform in platforms:
                    if platform not in self.profile_templates:
                        continue
                    
                    platform_config = self.profile_templates[platform]
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


# ================== NEW FUNCTIONS FOR URI FACE COMPARISON ==================

def compare_face_from_uri(face_system, uri: str, username: str = None, save_to_db: bool = False):
    """Compare a face from a URI (URL or local path) with indexed faces."""
    print(f"\n🔍 Comparing face from URI: {uri}")
    
    # Load target image
    target_bytes = get_image_bytes(uri)
    if not target_bytes:
        print("❌ Could not load image from URI")
        return
    
    # Check if it's a valid image
    try:
        Image.open(BytesIO(target_bytes)).verify()
    except Exception:
        print("❌ Invalid image file")
        return
    
    # Extract face encoding
    print("🧬 Extracting face encoding...")
    target_encoding = compute_face_encoding(target_bytes)
    if target_encoding is None:
        print("❌ No face detected in the image")
        return
    
    print("✅ Face encoding extracted successfully")
    
    # Get comparison parameters
    try:
        threshold = float(input("Match threshold (0.1-1.0, default 0.6): ") or "0.6")
        top_k = int(input("Number of results to show (default 20): ") or "20")
    except ValueError:
        threshold = 0.6
        top_k = 20
    
    # Search for matches
    print(f"\n🔍 Searching {len(face_system.faces)} indexed faces...")
    matches = face_system.search_faces(target_encoding, threshold, top_k)
    
    if not matches:
        print("❌ No matches found")
        return matches
    
    # Display results
    print(f"\n🏆 Top {len(matches)} matches:")
    for i, match in enumerate(matches, 1):
        symbol = "✅" if match["match"] else "⚠️"
        similarity_percent = match['similarity'] * 100
        
        # Color code based on similarity
        if similarity_percent >= 80:
            similarity_str = f"🎯 {similarity_percent:.1f}%"
        elif similarity_percent >= 60:
            similarity_str = f"🔍 {similarity_percent:.1f}%"
        else:
            similarity_str = f"📊 {similarity_percent:.1f}%"
        
        print(f"\n  {i}. {symbol} {similarity_str}")
        print(f"     User: {match['username']}")
        print(f"     Platform: {match['platform']}")
        print(f"     Distance: {match['distance']:.4f}")
        
        if match['image_url']:
            print(f"     Image: {match['image_url'][:80]}...")
    
    # Option to save the face to database
    if save_to_db and username:
        save_face_to_db(face_system, target_encoding, uri, username, uri)
        print(f"✅ Face saved to database with username: {username}")
    
    return matches


def save_face_to_db(face_system, encoding: np.ndarray, image_url: str, username: str, page_url: str = None, platform: str = "direct_uri"):
    """Save a face to the database directly."""
    face_record = {
        "username": username,
        "platform": platform,
        "page_url": page_url or image_url,
        "image_url": image_url,
        "encoding": encoding.tolist(),
        "timestamp": time.time(),
        "source": "direct_uri"
    }
    
    face_system.faces.append(face_record)
    return face_record


def batch_compare_from_file(face_system, filename: str):
    """Compare faces from a file containing URIs and usernames."""
    if not os.path.exists(filename):
        print(f"❌ File '{filename}' not found")
        return
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    print(f"\n📄 Processing {len(lines)} entries from {filename}...")
    
    results = []
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split(',')
        if len(parts) >= 2:
            uri = parts[0].strip()
            username = parts[1].strip()
            
            print(f"\n[{line_num}] Processing {username} - {uri}")
            
            # Get the face
            target_bytes = get_image_bytes(uri)
            if not target_bytes:
                print(f"  ❌ Could not load image")
                continue
            
            target_encoding = compute_face_encoding(target_bytes)
            if target_encoding is None:
                print(f"  ❌ No face detected")
                continue
            
            # Search for matches
            matches = face_system.search_faces(target_encoding, threshold=0.6, top_k=5)
            
            if matches:
                best_match = matches[0]
                results.append({
                    'uri': uri,
                    'username': username,
                    'best_match': best_match['username'],
                    'similarity': best_match['similarity'],
                    'platform': best_match['platform']
                })
                
                print(f"  🔍 Best match: {best_match['username']} ({best_match['similarity']:.3f})")
            else:
                print(f"  ⚠️ No matches found")
    
    # Save results
    if results:
        output_file = f"comparison_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")
    
    return results


def extract_faces_from_webpage(url: str, username: str = None):
    """Extract faces from a webpage URL."""
    print(f"\n🌐 Extracting faces from webpage: {url}")
    
    crawler = EnhancedProfileCrawler()
    
    # Get the page
    try:
        response = requests.get(
            url,
            headers={'User-Agent': UserAgent().random},
            timeout=15
        )
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Error fetching webpage: {e}")
        return []
    
    # Extract images
    image_urls = crawler.extract_images(response.text, url, {})
    
    if not image_urls:
        print("❌ No images found on page")
        return []
    
    print(f"📸 Found {len(image_urls)} images")
    
    faces = []
    for i, img_url in enumerate(image_urls[:10], 1):  # Limit to first 10 images
        print(f"  [{i}] Processing: {img_url[:80]}...")
        
        img_bytes = get_image_bytes(img_url)
        if not img_bytes:
            continue
        
        encoding = compute_face_encoding(img_bytes)
        if encoding is not None:
            faces.append({
                'image_url': img_url,
                'encoding': encoding,
                'page_url': url
            })
            print(f"    ✅ Face detected")
    
    print(f"\n✅ Found {len(faces)} faces on the webpage")
    return faces


def create_uri_batch_file():
    """Create a template batch file for URI comparisons."""
    template = """# URI comparison batch file
# Format: image_url_or_path,username,optional_platform
# 
# Examples:
https://example.com/face1.jpg,john_doe,facebook
https://example.com/face2.jpg,jane_smith,instagram
/path/to/local/image.jpg,anonymous,direct
"""
    
    filename = f"uri_batch_{int(time.time())}.txt"
    with open(filename, 'w') as f:
        f.write(template)
    
    print(f"📝 Created batch template file: {filename}")
    print("Edit this file with your URIs and usernames, then use option 7.")


# ================== NEW FUNCTION: UPLOAD IMAGE AND SEARCH PLATFORMS ==================

def search_platforms_by_face(face_system, crawler):
    """Search selected platforms using a face image."""
    print("\n📸 Upload Image and Search Platforms")
    print("-" * 40)
    
    # Get image
    uri = input("Enter image path or URL: ").strip()
    if not uri:
        print("❌ No image provided")
        return
    
    # Extract face from image
    print("🔍 Extracting face from image...")
    target_bytes = get_image_bytes(uri)
    if not target_bytes:
        print("❌ Could not load image")
        return
    
    target_encoding = compute_face_encoding(target_bytes)
    if target_encoding is None:
        print("❌ No face detected in image")
        return
    
    print("✅ Face encoding extracted")
    
    # Ask if they want to search with a specific username
    use_username = input("\nDo you want to search with a specific username? (y/N): ").strip().lower()
    username_to_search = None
    
    if use_username == 'y':
        username_to_search = input("Enter username to search: ").strip()
        if username_to_search:
            print(f"🔍 Will search for username: {username_to_search}")
    
    # Show available platforms
    templates = load_profile_templates()
    enabled_platforms = get_enabled_platforms(templates)
    categories = get_platforms_by_category(templates)
    
    if not enabled_platforms:
        print("❌ No platforms enabled in profile_templates.json")
        return
    
    print(f"\n📋 Available platforms ({len(enabled_platforms)} enabled):")
    for category, platforms in categories.items():
        print(f"\n  {category}:")
        for platform in sorted(platforms):
            config = templates[platform]
            url_template = config.get("url", "No URL")
            print(f"    {platform:20} - {url_template}")
    
    # Select platforms
    platform_input = input("\nEnter platforms to search (comma-separated, or 'all'): ").strip().lower()
    
    if platform_input == 'all':
        selected_platforms = enabled_platforms
    else:
        selected_platforms = []
        for item in platform_input.split(','):
            item = item.strip()
            if item in enabled_platforms:
                selected_platforms.append(item)
    
    if not selected_platforms:
        print("⚠️  No platforms selected")
        return
    
    print(f"\n🔍 Will search {len(selected_platforms)} platform(s)")
    
    # Search logic
    results = []
    
    if username_to_search:
        # Search specific username on selected platforms
        print(f"\n🔎 Searching username '{username_to_search}' on {len(selected_platforms)} platforms...")
        
        crawl_results = crawler.crawl_usernames([username_to_search], selected_platforms)
        
        # Check if any profiles exist
        user_results = crawl_results.get(username_to_search, [])
        existing_profiles = [r for r in user_results if r["exists"]]
        
        if existing_profiles:
            print(f"\n✅ Found {len(existing_profiles)} profile(s) for '{username_to_search}':")
            
            for result in existing_profiles:
                print(f"\n  Platform: {result['platform']}")
                print(f"  URL: {result['url']}")
                print(f"  Images found: {len(result['image_urls'])}")
                
                # Compare face with images from profile
                if result["image_urls"]:
                    print("  Comparing faces from profile images...")
                    
                    best_similarity = 0
                    best_match_url = None
                    
                    for img_url in result["image_urls"][:3]:  # Check first 3 images
                        try:
                            img_bytes = get_image_bytes(img_url)
                            if img_bytes:
                                img_encoding = compute_face_encoding(img_bytes)
                                if img_encoding is not None:
                                    distance = float(face_recognition.face_distance([target_encoding], img_encoding)[0])
                                    similarity = max(0.0, 1.0 - min(distance, 1.0))
                                    
                                    if similarity > best_similarity:
                                        best_similarity = similarity
                                        best_match_url = img_url
                        except Exception:
                            continue
                    
                    if best_similarity > 0:
                        print(f"  🎯 Best face match: {best_similarity:.3f}")
                        if best_similarity > 0.6:
                            print(f"  ✅ LIKELY SAME PERSON!")
                        
                        results.append({
                            "platform": result["platform"],
                            "url": result["url"],
                            "username": username_to_search,
                            "similarity": best_similarity,
                            "match_url": best_match_url,
                            "type": "username_search"
                        })
                else:
                    print("  ⚠️ No images to compare")
        else:
            print(f"❌ No profiles found for '{username_to_search}' on selected platforms")
    
    else:
        # First check existing face database
        print("\n🔍 Checking existing face database...")
        matches = face_system.search_faces(target_encoding, threshold=0.6, top_k=5)
        
        if matches:
            print(f"🎯 Found {len([m for m in matches if m['match']])} potential matches in database:")
            for match in matches[:3]:  # Show top 3
                if match['match']:
                    print(f"  👤 {match['username']} on {match['platform']} - similarity: {match['similarity']:.3f}")
            
            # Ask if they want to search for these usernames
            search_existing = input("\nSearch platforms for these usernames? (y/N): ").strip().lower()
            
            if search_existing == 'y':
                usernames_to_search = list(set([m['username'] for m in matches if m['match']]))[:5]
                print(f"🔍 Searching for {len(usernames_to_search)} username(s): {', '.join(usernames_to_search)}")
                
                crawl_results = crawler.crawl_usernames(usernames_to_search, selected_platforms)
                
                for username in usernames_to_search:
                    user_results = crawl_results.get(username, [])
                    for result in user_results:
                        if result["exists"]:
                            results.append({
                                "platform": result["platform"],
                                "url": result["url"],
                                "username": username,
                                "type": "database_match_search"
                            })
        
        # Option to do a reverse image search style lookup
        print("\n🔄 Alternative: Check popular platforms for matching faces")
        print("   (This will crawl and compare faces from profiles)")
        
        do_crawl = input("\nCrawl and compare faces from profiles? (y/N): ").strip().lower()
        
        if do_crawl == 'y':
            # For demo, we'd need to implement a broader search
            # For now, we'll show a message
            print("\n⚠️  Advanced face-based platform crawling would require:")
            print("   1. Generating common usernames from face similarity")
            print("   2. Searching those usernames on platforms")
            print("   3. Comparing faces from found profiles")
            print("\n💡 Tip: Use option 1 with suspected usernames first")
    
    # Display results
    if results:
        print(f"\n📊 Search Results ({len(results)}):")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Platform: {result['platform']}")
            print(f"   Username: {result['username']}")
            print(f"   URL: {result['url']}")
            print(f"   Type: {result['type']}")
            
            if 'similarity' in result:
                print(f"   Face Similarity: {result['similarity']:.3f}")
                if result['similarity'] > 0.7:
                    print("   🎯 HIGH CONFIDENCE MATCH")
                elif result['similarity'] > 0.5:
                    print("   🔍 Possible match")
            
            if 'match_url' in result:
                print(f"   Match Image: {result['match_url'][:80]}...")
    else:
        print("\n❌ No results found")
    
    # Offer to save the face to database
    if target_encoding is not None:
        save_face = input("\n💾 Save this face to database for future searches? (y/N): ").strip().lower()
        if save_face == 'y':
            username = input("Enter username for this face (or leave blank for 'unknown'): ").strip() or "unknown"
            platform = input("Enter source platform (or leave blank for 'upload'): ").strip() or "upload"
            
            face_record = {
                "username": username,
                "platform": platform,
                "page_url": uri if uri.startswith('http') else f"file://{os.path.abspath(uri)}",
                "image_url": uri,
                "encoding": target_encoding.tolist(),
                "timestamp": time.time(),
                "source": "image_upload_search"
            }
            
            face_system.faces.append(face_record)
            print(f"✅ Face saved to database as '{username}'")


# ================== TEMPLATE MANAGEMENT FUNCTIONS ==================

def manage_templates_menu():
    """Menu for managing profile templates."""
    while True:
        print("\n" + "=" * 60)
        print("Profile Templates Management")
        print("=" * 60)
        print("1. List all platforms")
        print("2. List by category")
        print("3. Add new platform")
        print("4. Edit existing platform")
        print("5. Enable/Disable platform")
        print("6. Export templates to JSON")
        print("7. Import templates from JSON")
        print("8. Back to main menu")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == "1":
            # List all platforms
            templates = load_profile_templates()
            print(f"\n📋 All Platforms ({len(templates)} total):")
            for i, (platform_name, config) in enumerate(templates.items(), 1):
                enabled = "✅" if config.get("enabled", True) else "❌"
                category = config.get("category", "Uncategorized")
                print(f"  {i:2d}. {enabled} {platform_name:20} - {category}")
        
        elif choice == "2":
            # List by category
            templates = load_profile_templates()
            categories = get_platforms_by_category(templates)
            
            print("\n📋 Platforms by Category:")
            for category, platforms in categories.items():
                print(f"\n  {category}:")
                for platform in sorted(platforms):
                    config = templates[platform]
                    enabled = "✅" if config.get("enabled", True) else "❌"
                    url_template = config.get("url", "No URL")
                    print(f"      {enabled} {platform:20} - {url_template}")
        
        elif choice == "3":
            # Add new platform
            print("\n➕ Add New Platform")
            
            platform_name = input("Platform name (lowercase, no spaces): ").strip().lower()
            if not platform_name:
                print("❌ Platform name required")
                continue
            
            templates = load_profile_templates()
            if platform_name in templates:
                print(f"❌ Platform '{platform_name}' already exists")
                continue
            
            url_template = input("URL template (use {} for username): ").strip()
            if not url_template or "{}" not in url_template:
                print("❌ URL template must contain {} placeholder for username")
                continue
            
            category = input("Category (e.g., Social Media, Tech, etc): ").strip() or "Other"
            
            print("\nAvailable check methods:")
            print("  status_code - Simple 200 OK check")
            print("  universal_check - Universal profile check (recommended)")
            print("  github_check - GitHub specific check")
            print("  twitter_check - Twitter specific check")
            print("  instagram_check - Instagram specific check")
            print("  fansfinder_check - OnlyFans via FansFinder check")
            
            check_method = input("Check method (default: universal_check): ").strip() or "universal_check"
            
            avatar_selector = input("Avatar CSS selector (optional): ").strip()
            
            requires_js = input("Requires JavaScript? (y/N): ").strip().lower() == 'y'
            
            new_template = {
                "url": url_template,
                "check_method": check_method,
                "avatar_selector": avatar_selector,
                "requires_javascript": requires_js,
                "platform": platform_name,
                "category": category,
                "enabled": True,
                "priority": 3
            }
            
            templates[platform_name] = new_template
            save_profile_templates(templates)
            print(f"✅ Platform '{platform_name}' added successfully")
        
        elif choice == "4":
            # Edit existing platform
            templates = load_profile_templates()
            
            print("\n✏️  Edit Platform")
            platforms = list(templates.keys())
            
            for i, platform in enumerate(platforms, 1):
                print(f"  {i:2d}. {platform}")
            
            try:
                selection = int(input("\nSelect platform number: ").strip())
                if 1 <= selection <= len(platforms):
                    platform_name = platforms[selection - 1]
                    config = templates[platform_name]
                    
                    print(f"\nEditing: {platform_name}")
                    print(f"Current URL: {config.get('url')}")
                    new_url = input(f"New URL (Enter to keep current): ").strip()
                    if new_url:
                        if "{}" not in new_url:
                            print("❌ URL must contain {} placeholder")
                            continue
                        config["url"] = new_url
                    
                    print(f"Current category: {config.get('category')}")
                    new_category = input(f"New category: ").strip()
                    if new_category:
                        config["category"] = new_category
                    
                    print(f"Current check method: {config.get('check_method')}")
                    new_check = input(f"New check method: ").strip()
                    if new_check:
                        config["check_method"] = new_check
                    
                    print(f"Current avatar selector: {config.get('avatar_selector')}")
                    new_selector = input(f"New avatar selector: ").strip()
                    if new_selector:
                        config["avatar_selector"] = new_selector
                    
                    save_profile_templates(templates)
                    print(f"✅ Platform '{platform_name}' updated")
                else:
                    print("❌ Invalid selection")
            except (ValueError, IndexError):
                print("❌ Invalid input")
        
        elif choice == "5":
            # Enable/Disable platform
            templates = load_profile_templates()
            
            print("\n⚙️  Enable/Disable Platform")
            platforms = list(templates.keys())
            
            for i, platform in enumerate(platforms, 1):
                enabled = "✅" if templates[platform].get("enabled", True) else "❌"
                print(f"  {i:2d}. {enabled} {platform}")
            
            try:
                selection = int(input("\nSelect platform number: ").strip())
                if 1 <= selection <= len(platforms):
                    platform_name = platforms[selection - 1]
                    current = templates[platform_name].get("enabled", True)
                    templates[platform_name]["enabled"] = not current
                    
                    status = "enabled" if not current else "disabled"
                    save_profile_templates(templates)
                    print(f"✅ Platform '{platform_name}' {status}")
                else:
                    print("❌ Invalid selection")
            except (ValueError, IndexError):
                print("❌ Invalid input")
        
        elif choice == "6":
            # Export templates
            filename = input("Export filename (default: profile_templates_export.json): ").strip() or "profile_templates_export.json"
            templates = load_profile_templates()
            save_profile_templates(templates, filename)
            print(f"✅ Templates exported to {filename}")
        
        elif choice == "7":
            # Import templates
            filename = input("Import filename: ").strip()
            if not filename:
                print("❌ Filename required")
                continue
            
            if not os.path.exists(filename):
                print(f"❌ File '{filename}' not found")
                continue
            
            try:
                with open(filename, 'r') as f:
                    imported = json.load(f)
                
                # Merge or replace?
                print("\nImport options:")
                print("  1. Merge with existing (keep both)")
                print("  2. Replace existing (overwrite)")
                print("  3. Cancel")
                
                option = input("Select option (1-3): ").strip()
                
                if option == "1":
                    templates = load_profile_templates()
                    templates.update(imported)
                    save_profile_templates(templates)
                    print(f"✅ Merged {len(imported)} templates")
                elif option == "2":
                    save_profile_templates(imported)
                    print(f"✅ Replaced with {len(imported)} templates")
                else:
                    print("❌ Import cancelled")
            
            except Exception as e:
                print(f"❌ Error importing: {e}")
        
        elif choice == "8":
            # Back to main menu
            break


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
        print("2. Upload image and search selected platforms (NEW)")
        print("3. Test specific profile")
        print("4. Run known profile tests")
        print("5. Compare target face (from local image)")
        print("6. Compare face from URL/URI")
        print("7. Extract faces from webpage")
        print("8. Batch compare from file")
        print("9. Create batch template")
        print("10. Show statistics")
        print("11. Manage profile templates")
        print("12. Save face index")
        print("13. Load face index")
        print("14. Clear face index")
        print("15. Exit")
        
        choice = input("\nSelect option (1-15): ").strip()
        
        if choice == "1":
            # Search usernames (existing code)
            usernames_input = input("Enter usernames (comma-separated): ").strip()
            if not usernames_input:
                continue
            
            usernames = [u.strip() for u in usernames_input.split(',')]
            
            # Platform selection
            templates = load_profile_templates()
            enabled_platforms = get_enabled_platforms(templates)
            categories = get_platforms_by_category(templates)
            
            print(f"\n📋 Available platforms ({len(enabled_platforms)} enabled):")
            for category, platforms in categories.items():
                print(f"\n  {category}:")
                for platform in sorted(platforms):
                    config = templates[platform]
                    url_template = config.get("url", "No URL")
                    print(f"    {platform:20} - {url_template}")
            
            platform_input = input("\nEnter platforms (comma-separated, or 'all'): ").strip().lower()
            
            if platform_input == 'all':
                selected_platforms = enabled_platforms
            else:
                selected_platforms = []
                for item in platform_input.split(','):
                    item = item.strip()
                    if item in enabled_platforms:
                        selected_platforms.append(item)
            
            if not selected_platforms:
                # Default to all enabled platforms from the JSON file
                selected_platforms = enabled_platforms
                print(f"⚠️  No platforms specified, using all {len(selected_platforms)} enabled platforms")
            
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
                save = input("\n💾 Save results to face index? (y/N): ").strip().lower()
                if save == 'y':
                    face_system.save_index()
        
        elif choice == "2":
            # NEW: Upload image and search selected platforms
            search_platforms_by_face(face_system, crawler)
        
        elif choice == "3":
            test_specific_profile()
        
        elif choice == "4":
            test_known_profiles()
        
        elif choice == "5":
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
        
        elif choice == "6":
            # Compare face from URL/URI
            uri = input("Enter image URL or local file path: ").strip()
            if not uri:
                continue
            
            print("\nOptions:")
            print("1. Just compare with existing faces")
            print("2. Compare and save to database")
            
            sub_choice = input("Select (1-2): ").strip()
            
            if sub_choice == "2":
                username = input("Enter username for this face: ").strip()
                if username:
                    compare_face_from_uri(face_system, uri, username, save_to_db=True)
                else:
                    print("❌ Username required to save to database")
            else:
                compare_face_from_uri(face_system, uri)
        
        elif choice == "7":
            # Extract faces from webpage
            url = input("Enter webpage URL: ").strip()
            if not url:
                continue
            
            faces = extract_faces_from_webpage(url)
            
            if faces:
                print("\nOptions:")
                print("1. Compare each face with database")
                print("2. Save all faces to database")
                
                sub_choice = input("Select (1-2): ").strip()
                
                if sub_choice == "1":
                    for i, face in enumerate(faces, 1):
                        print(f"\n[{i}] Comparing face from image...")
                        temp_uri = face['image_url']
                        matches = compare_face_from_uri(face_system, temp_uri)
                        
                        if matches and len(matches) > 0:
                            best = matches[0]
                            if best['similarity'] > 0.7:
                                save = input(f"  Save as match to {best['username']}? (y/N): ").strip().lower()
                                if save == 'y':
                                    username = input(f"  Username (default: {best['username']}): ").strip() or best['username']
                                    save_face_to_db(face_system, face['encoding'], face['image_url'], username, face['page_url'], "webpage_extraction")
                                    print(f"  ✅ Saved to database")
                
                elif sub_choice == "2":
                    username = input("Base username (faces will be saved as username_1, username_2, etc): ").strip()
                    if username:
                        for i, face in enumerate(faces, 1):
                            user_id = f"{username}_{i}"
                            save_face_to_db(face_system, face['encoding'], face['image_url'], user_id, face['page_url'], "webpage_extraction")
                        print(f"✅ Saved {len(faces)} faces to database")
                    else:
                        print("❌ Username required")
        
        elif choice == "8":
            # Batch compare from file
            filename = input("Enter filename with URIs and usernames (CSV format): ").strip()
            if filename and os.path.exists(filename):
                batch_compare_from_file(face_system, filename)
            else:
                print("❌ File not found")
        
        elif choice == "9":
            # Create batch template
            create_uri_batch_file()
        
        elif choice == "10":
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
                
                # Count by source
                sources = {}
                for face in face_system.faces:
                    source = face.get("source", "unknown")
                    sources[source] = sources.get(source, 0) + 1
                
                print(f"  By source:")
                for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {source}: {count}")
        
        elif choice == "11":
            manage_templates_menu()
        
        elif choice == "12":
            filename = input("Filename (default: face_index.json): ").strip() or "face_index.json"
            face_system.save_index(filename)
        
        elif choice == "13":
            filename = input("Filename (default: face_index.json): ").strip() or "face_index.json"
            face_system.load_index(filename)
        
        elif choice == "14":
            confirm = input("Clear all indexed faces? (y/N): ").strip().lower()
            if confirm == 'y':
                face_system.faces = []
                print("✅ Face index cleared")
        
        elif choice == "15":
            print("👋 Goodbye!")
            break


if __name__ == "__main__":
    main()