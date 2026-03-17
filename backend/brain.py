import requests
import json
import concurrent.futures

class NeuralDirector:
    """Глобальный разум колонии, работающий через Ollama."""
    def __init__(self, model_name="phi3"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/generate"
        self.latest_directive = "Explore and gather resources."
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.is_thinking = False

    def _fetch_ollama(self, prompt):
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=3
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except:
            return None
        return None

    def analyze_and_direct(self, stats):
        """Анализирует статистику колонии (неблокирующий вызов)."""
        if self.is_thinking:
            return self.latest_directive

        prompt = f"Stats: {stats['robot_count']} robots, {stats['resource_count']} resources, {stats['avg_energy']}% energy. Give 5-word command."
        
        def callback(future):
            self.is_thinking = False
            res = future.result()
            if res:
                self.latest_directive = res
            else:
                # Fallback logic
                if stats['avg_energy'] < 40:
                    self.latest_directive = "CRITICAL: Low Energy!"
                else:
                    self.latest_directive = "Nominal operation."

        self.is_thinking = True
        future = self.executor.submit(self._fetch_ollama, prompt)
        future.add_done_callback(callback)
        
        return self.latest_directive
