## Auto-Deploy Setup
1. Create a GCP service account with roles: Cloud Run Developer, Cloud Build Editor, Storage Admin
2. Download the JSON key
3. Add it as GitHub secret: Settings → Secrets → New secret → name: GCP_SA_KEY, value: (JSON content)
4. Push to main to trigger deployment
