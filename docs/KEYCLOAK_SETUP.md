# Keycloak Setup Guide

This guide provides step-by-step instructions for configuring Keycloak to work with the Marketing Tool application.

## Prerequisites

- Keycloak server deployed and accessible
- Admin access to Keycloak administration console
- Basic understanding of OAuth2/OIDC concepts

## Step 1: Create a Realm

1. Log in to the Keycloak administration console
2. Click on the realm dropdown (top left) and select "Create Realm"
3. Enter a realm name (e.g., `marketing-tool`)
4. Click "Create"

## Step 2: Create Backend Client (Confidential)

The backend client is used by the FastAPI server to validate JWT tokens.

1. In your realm, navigate to **Clients** → **Create client**
2. Configure the client:
   - **Client type**: OpenID Connect
   - **Client ID**: `marketing-tool-backend`
   - Click **Next**

3. Configure capabilities:
   - **Client authentication**: ON (this makes it a confidential client)
   - **Authorization**: OFF (unless you need fine-grained authorization)
   - **Standard flow**: ON
   - Click **Next**

4. Configure login settings:
   - **Root URL**: Leave empty or set to your backend URL
   - **Home URL**: Leave empty
   - **Valid redirect URIs**: Leave empty (backend doesn't use redirects)
   - **Web origins**: Add your backend URL (e.g., `http://localhost:8000`)
   - Click **Save**

5. After saving, go to the **Credentials** tab and copy the **Client secret**

6. Configure token settings (optional but recommended):
   - Go to **Advanced settings**
   - Set **Access Token Lifespan** to your preferred value (default: 5 minutes)
   - Set **SSO Session Idle** to your preferred value (default: 30 minutes)

## Step 3: Create Frontend Client (Public)

The frontend client is used by the Next.js application for user authentication.

1. Navigate to **Clients** → **Create client**
2. Configure the client:
   - **Client type**: OpenID Connect
   - **Client ID**: `marketing-tool-frontend`
   - Click **Next**

3. Configure capabilities:
   - **Client authentication**: OFF (this makes it a public client)
   - **Authorization**: OFF
   - **Standard flow**: ON
   - **Direct access grants**: OFF (unless needed)
   - Click **Next**

4. Configure login settings:
   - **Root URL**: Your frontend URL (e.g., `http://localhost:3000`)
   - **Home URL**: Your frontend URL
   - **Valid redirect URIs**:
     - `http://localhost:3000/api/auth/callback/keycloak` (development)
     - `https://yourdomain.com/api/auth/callback/keycloak` (production)
   - **Web origins**:
     - `http://localhost:3000` (development)
     - `https://yourdomain.com` (production)
   - Click **Save**

5. After saving, go to the **Credentials** tab and copy the **Client secret** (even though it's a public client, NextAuth may need it)

## Step 4: Create Roles

1. Navigate to **Realm roles** → **Create role**
2. Create the following roles:
   - `admin` - Full access to all features
   - `editor` - Can create and edit content
   - `user` - Basic access, can view content

3. Optionally, you can also create client-specific roles:
   - Navigate to **Clients** → Select your backend client → **Roles** tab
   - Create client-specific roles if needed

## Step 5: Assign Roles to Users

1. Navigate to **Users** → Select a user (or create a new user)
2. Go to the **Role mapping** tab
3. Click **Assign role**
4. Select the roles you want to assign (e.g., `admin`, `editor`, `user`)
5. Click **Assign**

## Step 6: Configure Token Mapper (Optional)

To ensure roles are included in the JWT token:

1. Navigate to **Clients** → Select your backend client → **Client scopes**
2. Click on the client scope (usually named after your client)
3. Go to the **Mappers** tab
4. If not present, add a mapper:
   - Click **Add mapper** → **By configuration**
   - Select **Realm roles** mapper
   - Configure:
     - **Name**: `realm-roles`
     - **Token Claim Name**: `realm_access.roles`
     - **Add to access token**: ON
     - **Add to ID token**: ON
     - Click **Save**

5. Similarly, add a client roles mapper if using client-specific roles

## Step 7: Get Realm Public Key (Optional)

If you want to use a static public key instead of fetching it dynamically:

1. Navigate to **Realm settings** → **Keys** tab
2. Find the **RS256** key
3. Click on the key to view details
4. Copy the **Public key** (PEM format)
5. Set this in your backend environment variable: `KEYCLOAK_PUBLIC_KEY`

## Step 8: Configure Environment Variables

### Backend (.env)

```bash
KEYCLOAK_SERVER_URL=https://your-keycloak-server.com
KEYCLOAK_REALM=marketing-tool
KEYCLOAK_CLIENT_ID=marketing-tool-backend
KEYCLOAK_CLIENT_SECRET=your-backend-client-secret
KEYCLOAK_PUBLIC_KEY=  # Optional: public key in PEM format
KEYCLOAK_VERIFY_SSL=true
```

### Frontend (.env.local)

```bash
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-nextauth-secret-here  # Generate with: openssl rand -base64 32
KEYCLOAK_CLIENT_ID=marketing-tool-frontend
KEYCLOAK_CLIENT_SECRET=your-frontend-client-secret
KEYCLOAK_ISSUER=https://your-keycloak-server.com/realms/marketing-tool
```

## Step 9: Test the Configuration

1. Start your backend server
2. Start your frontend application
3. Navigate to the login page
4. You should be redirected to Keycloak login
5. After logging in, you should be redirected back to the application
6. Check that your user roles are displayed correctly

## Troubleshooting

### Token Validation Fails

- Verify the `KEYCLOAK_SERVER_URL` and `KEYCLOAK_REALM` are correct
- Check that the public key is being fetched correctly (check backend logs)
- Ensure the token issuer matches: `{KEYCLOAK_SERVER_URL}/realms/{KEYCLOAK_REALM}`

### Roles Not Appearing

- Verify roles are assigned to the user in Keycloak
- Check that the token mapper is configured correctly
- Ensure roles are included in the access token (check token contents)

### Redirect URI Mismatch

- Verify the redirect URI in Keycloak matches exactly: `{NEXTAUTH_URL}/api/auth/callback/keycloak`
- Check for trailing slashes and protocol (http vs https)

### CORS Issues

- Ensure the frontend URL is added to **Web origins** in Keycloak client settings
- Check backend CORS configuration allows the frontend origin

## Security Best Practices

1. **Use HTTPS in production** - Never use HTTP for Keycloak in production
2. **Rotate secrets regularly** - Change client secrets periodically
3. **Set appropriate token lifespans** - Don't make tokens too long-lived
4. **Use role-based access control** - Assign minimal necessary roles
5. **Enable SSL verification** - Keep `KEYCLOAK_VERIFY_SSL=true` in production
6. **Protect client secrets** - Never commit secrets to version control

## Additional Resources

- [Keycloak Documentation](https://www.keycloak.org/documentation)
- [NextAuth.js Documentation](https://next-auth.js.org/)
- [OAuth2/OIDC Specification](https://oauth.net/2/)
