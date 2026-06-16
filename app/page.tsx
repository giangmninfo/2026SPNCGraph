"use client"
import { createUser } from "@/lib/api/users"
import { login, validateToken } from "@/lib/api/auth"

import { ImageClassifier } from "@/components/image-classifier"
import { ThemeToggle } from "@/components/theme-toggle"
import { Button } from "@/components/ui/button"
import { useState, useEffect } from "react"
import { AuthModals } from "@/components/auth-modals"
import { ProfileModal } from "@/components/profile-modal"
import { useNotification, NotificationContainer } from "@/components/notification"
import { GitHubAvatar } from "@/components/github-avatar"

export default function Home() {
  const { notifications, addNotification, removeNotification } = useNotification()
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [showSignIn, setShowSignIn] = useState(false)
  const [showSignUp, setShowSignUp] = useState(false)
  const [showProfile, setShowProfile] = useState(false)
  const [authChecked, setAuthChecked] = useState(false)
  const [user, setUser] = useState({
    fullName: "",
    username: "",
    email: "",
  })
  const [logoutSignal, setLogoutSignal] = useState(0)

  useEffect(() => {
    const bootAuthCheck = async () => {
      try {
        await validateToken()
  
        const storedUser = localStorage.getItem("currentUser")
        if (!storedUser) throw new Error()
  
        const parsed = JSON.parse(storedUser)
  
        setUser({
          fullName: parsed.fullName,
          username: parsed.username,
          email: parsed.email,
        })
  
        setIsAuthenticated(true)
      } catch {
        handleLogout()
      } finally {
        setAuthChecked(true)
      }
    }
  
    bootAuthCheck()
  }, [])

  const handleSignIn = async (identifier: string, password: string) => {
    try {
      const data = await login({ identifier, password })

      localStorage.setItem("access_token", data.access_token)
      localStorage.setItem("currentUser", JSON.stringify({
        fullName: data.user.full_name,
        username: data.user.username,
        email: data.user.email,
      }))

      setUser({
        fullName: data.user.full_name, // unused
        username: data.user.username,
        email: data.user.email,
      })

      setIsAuthenticated(true)
      setShowSignIn(false)

      addNotification("Logged in successfully!", "success")
    } catch (err) {
      addNotification(
        err instanceof Error ? err.message : "Login failed",
        "error"
      )
    }
  }

  const handleSignUp = async (userData: { fullName: string, username: string, email: string, avatarColor: string, password: string }) => {
    try {
      await createUser({
        full_name: userData.fullName,
        username: userData.username,
        email: userData.email,
        password: userData.password,
        avatar_color: userData.avatarColor,
      })

      setUser({
        fullName: userData.fullName,
        username: userData.username,
        email: userData.email,
      })

      setIsAuthenticated(true)
      setShowSignUp(false)
      setShowSignIn(false)

      addNotification("Account created successfully!", "success")
    } catch (err: any) {
      addNotification(err.message || "Signup failed", "error")
    }
  }

  const handleLogout = () => {
    // localStorage.removeItem("currentUser")
    // localStorage.removeItem("userAvatarColor")
    localStorage.removeItem("access_token")
    localStorage.removeItem("currentUser")
    setIsAuthenticated(false)
    setShowProfile(false)
    setUser({ fullName: "", username: "", email: "" })
    setLogoutSignal((s) => s + 1)
    addNotification("Logged out successfully", "success")
  }

  return (
    <div className="min-h-screen bg-background">
      <NotificationContainer notifications={notifications} onRemove={removeNotification} />

      <header className="border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <button className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity">
            <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center">
              <span className="text-primary-foreground font-bold text-sm">
                <img
                  src="/logo.png"
                  alt="GNN Logo"
                  className="w-full h-full object-cover"
                />
              </span>
            </div>
            <h1 className="text-xl font-semibold text-foreground">Graph-Based Image Classifier</h1>
          </button>
          <div className="flex items-center gap-4">
            {!authChecked ? null : !isAuthenticated ? (
              <>
                <Button
                  variant="outline"
                  onClick={() => setShowSignIn(true)}
                >
                  Sign In
                </Button>

                <Button
                  variant="outline"
                  onClick={() => setShowSignUp(true)}
                >
                  Sign Up
                </Button>
              </>
            ) : (
              <button
                onClick={() => setShowProfile(true)}
                className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-accent/50 transition-colors"
              >
                <GitHubAvatar username={user.username} size={32} />
                <span className="text-sm font-medium text-foreground">{user.username}</span>
              </button>
            )}
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="text-center space-y-4">
            <h2 className="text-4xl md:text-5xl font-bold tracking-tight text-balance text-foreground">
              AI-Powered Image Classification
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto text-pretty">
              Upload an image and let our hybrid models identify the subject using advanced graph representation learning.
            </p>
          </div>

          <ImageClassifier isAuthenticated={isAuthenticated} onNotification={addNotification} resetSignal={logoutSignal} />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-8">
            <div className="p-6 rounded-lg border border-border bg-card">
              <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                <svg className="w-5 h-5 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2 text-foreground">Efficient Inference</h3>
              <p className="text-muted-foreground">
                Get classification results in seconds using optimized kNN search and GNN layers.
              </p>
            </div>

            <div className="p-6 rounded-lg border border-border bg-card">
              <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                <svg className="w-5 h-5 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2 text-foreground">High Precision</h3>
              <p className="text-muted-foreground">
                Leveraging local neighborhood structures and global graph features for precise detection.
              </p>
            </div>
          </div>

          <div className="p-6 rounded-lg border border-border bg-card/50 backdrop-blur-sm">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                <svg className="w-6 h-6 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C6.5 6.253 2 10.998 2 17s4.5 10.747 10 10.747c5.5 0 10-4.692 10-10.747S17.5 6.253 12 6.253z" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-lg mb-2 text-foreground">Built with Educational Excellence</h3>
                <p className="text-muted-foreground">
                  Our training dataset has been carefully curated and developed by our team through an educational program in Vietnam.
                  This collaborative effort ensures high-quality, diverse data that powers our model with real-world accuracy and reliability.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>

      <footer className="border-t border-border mt-20">
        <div className="container mx-auto px-4 py-8">
          <p className="text-center text-sm text-muted-foreground">
            © {new Date().getFullYear()} Le Hoang Anh's Team Classifier. All rights reserved.
          </p>
        </div>
      </footer>

      <AuthModals
        showSignIn={showSignIn}
        showSignUp={showSignUp}
        onSignInClose={() => setShowSignIn(false)}
        onSignUpClose={() => setShowSignUp(false)}
        onSignIn={handleSignIn}
        onSignUp={handleSignUp}
        onNotification={addNotification}
      />

      <ProfileModal
        isOpen={showProfile}
        onClose={() => setShowProfile(false)}
        onLogout={handleLogout}
        user={user}
      />
    </div>
  )
}
