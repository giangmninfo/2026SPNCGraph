'use client'

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { X, Check } from 'lucide-react'
import { PasswordInput } from "@/components/password-input"
import { useNotification } from "@/components/notification"
import { getAvatarColorForUser } from "@/components/github-avatar"
import { checkUsername } from "@/lib/api/users"

interface AuthModalsProps {
  onSignInClose: () => void
  onSignUpClose: () => void
  onOpenSignIn: () => void
  onSignIn: (email: string, password: string) => void
  onSignUp: (userData: { fullName: string, username: string, email: string, avatarColor: string, password: string }) => void
  showSignIn: boolean
  showSignUp: boolean
  onNotification: (message: string, type: 'success' | 'error' | 'info') => void
}

const validateEmail = (email: string) => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return emailRegex.test(email)
}

const validateUsername = (username: string) => {
  const usernameRegex = /^[a-z0-9]+$/
  return {
    valid: usernameRegex.test(username),
    message:
      "Username can only contain lowercase letters and numbers, with no spaces or special characters",
  }
}

// Password policy validation
const validatePassword = (password: string) => {
  return {
    length: password.length >= 9,
    uppercase: /[A-Z]/.test(password),
    lowercase: /[a-z]/.test(password),
    number: /[0-9]/.test(password),
    special: /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password),
  }
}

const isPasswordValid = (password: string) => {
  const policy = validatePassword(password)
  return policy.length && policy.uppercase && policy.lowercase && policy.number && policy.special
}

export function AuthModals({
  onSignInClose,
  onSignUpClose,
  onOpenSignIn,
  onSignIn,
  onSignUp,
  showSignIn,
  showSignUp,
  onNotification,
}: AuthModalsProps) {
  const [signInEmail, setSignInEmail] = useState("")
  const [signInPassword, setSignInPassword] = useState("")
  const [signUpFullName, setSignUpFullName] = useState("")
  const [signUpUsername, setSignUpUsername] = useState("")
  const [signUpEmail, setSignUpEmail] = useState("")
  const [signUpPassword, setSignUpPassword] = useState("")
  const [signUpConfirmPassword, setSignUpConfirmPassword] = useState("")
  const [showSignUpForm, setShowSignUpForm] = useState(false)
  const [usernameAvailable, setUsernameAvailable] = useState<boolean | null>(null)
  const [checkingUsername, setCheckingUsername] = useState(false)
  const [isSigningUp, setIsSigningUp] = useState(false)

  // useEffect(() => {
  //   if (signUpUsername.length > 0) {
  //     setCheckingUsername(true)
  //     const timer = setTimeout(() => {
  //       const users = JSON.parse(localStorage.getItem("users") || "[]")
  //       const exists = users.some((u: any) => u.username.toLowerCase() === signUpUsername.toLowerCase())
  //       setUsernameAvailable(!exists)
  //       setCheckingUsername(false)
  //     }, 500)
  //     return () => clearTimeout(timer)
  //   } else {
  //     setUsernameAvailable(null)
  //   }
  // }, [signUpUsername])

  useEffect(() => {
    // 🔴 RESET STATE WHEN INPUT IS EMPTY
    if (!signUpUsername) {
      setUsernameAvailable(null)
      setCheckingUsername(false)
      return
    }
  
    const { valid } = validateUsername(signUpUsername)
    if (!valid) {
      setUsernameAvailable(false)
      setCheckingUsername(false)
      return
    }
  
    setCheckingUsername(true)
  
    const timer = setTimeout(async () => {
      try {
        const res = await checkUsername(signUpUsername)
        setUsernameAvailable(res.available)
      } catch {
        setUsernameAvailable(null)
      } finally {
        setCheckingUsername(false)
      }
    }, 500)
  
    return () => clearTimeout(timer)
  }, [signUpUsername])
  

  const handleSignInSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!signInEmail || !signInPassword) {
      onNotification("Please fill in all fields", "error")
      return
    }
    onSignIn(signInEmail, signInPassword)
    setSignInEmail("")
    setSignInPassword("")
  }

  const handleSignUpSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!signUpFullName || !signUpUsername || !signUpEmail || !signUpPassword || !signUpConfirmPassword) {
      onNotification("Please fill in all fields", "error")
      return
    }

    const { valid, message } = validateUsername(signUpUsername)
    if (!valid) {
      onNotification(message, "error")
      return
    }

    if (!validateEmail(signUpEmail)) {
      onNotification("Please enter a valid email address", "error")
      return
    }

    if (!usernameAvailable) {
      onNotification("Username is not available or invalid", "error")
      return
    }

    if (!isPasswordValid(signUpPassword)) {
      onNotification("Password does not meet the requirements", "error")
      return
    }

    if (signUpPassword !== signUpConfirmPassword) {
      onNotification("Passwords do not match", "error")
      return
    }

    setIsSigningUp(true)
    
    const avatarColor = getAvatarColorForUser(signUpUsername)
    const userData = { fullName: signUpFullName, username: signUpUsername, email: signUpEmail, avatarColor, password: signUpPassword }
    
    onNotification("Sign up successful! Logging in...", "success")
    
    setTimeout(() => {
      onSignUp(userData)
      setSignUpFullName("")
      setSignUpUsername("")
      setSignUpEmail("")
      setSignUpPassword("")
      setSignUpConfirmPassword("")
      setShowSignUpForm(false)
      setIsSigningUp(false)
    }, 1500)
  }

  const passwordPolicy = validatePassword(signUpPassword)

  // Sign In Modal
  if (showSignIn && !showSignUpForm) {
    return (
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div className="modal-bg border border-border rounded-xl shadow-2xl w-full max-w-lg animate-in fade-in zoom-in-95">
          <div className="flex items-center justify-between p-6 border-b border-border">
            <h2 className="text-2xl font-semibold text-foreground">Sign In</h2>
            <button
              onClick={onSignInClose}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <form onSubmit={handleSignInSubmit} className="p-6 space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Email or Username</label>
              <Input
                placeholder="Enter your email or username"
                value={signInEmail}
                onChange={(e) => setSignInEmail(e.target.value)}
                required
                className="dark:bg-input dark:text-white dark:placeholder-muted-foreground bg-background text-foreground"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Password</label>
              <PasswordInput
                placeholder="Enter your password"
                value={signInPassword}
                onChange={(e) => setSignInPassword(e.target.value)}
                required
              />
            </div>

            <Button type="submit" className="w-full" size="lg">
              Sign In
            </Button>

            <div className="text-center text-sm">
              <span className="text-foreground">Don't have an account? </span>
              <button
                type="button"
                onClick={() => {
                  setShowSignUpForm(true)
                }}
                className="text-primary hover:underline font-medium"
              >
                Create one
              </button>
            </div>
          </form>
        </div>
      </div>
    )
  }

  if (showSignIn) {
<div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div className="modal-bg border border-border rounded-xl shadow-2xl w-full max-w-lg animate-in fade-in zoom-in-95">
          <div className="flex items-center justify-between p-6 border-b border-border">
            <h2 className="text-2xl font-semibold text-foreground">Sign In</h2>
            <button
              onClick={onSignInClose}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <form onSubmit={handleSignInSubmit} className="p-6 space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Email or Username</label>
              <Input
                placeholder="Enter your email or username"
                value={signInEmail}
                onChange={(e) => setSignInEmail(e.target.value)}
                required
                className="dark:bg-input dark:text-white dark:placeholder-muted-foreground bg-background text-foreground"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Password</label>
              <PasswordInput
                placeholder="Enter your password"
                value={signInPassword}
                onChange={(e) => setSignInPassword(e.target.value)}
                required
              />
            </div>

            <Button type="submit" className="w-full" size="lg">
              Sign In
            </Button>

            <div className="text-center text-sm">
              <span className="text-foreground">Don't have an account? </span>
              <button
                type="button"
                onClick={() => {
                  setShowSignUpForm(true)
                }}
                className="text-primary hover:underline font-medium"
              >
                Create one
              </button>
            </div>
          </form>
        </div>
      </div>
  }
  // Sign Up Modal
  if (showSignUp || showSignUpForm) {
    return (
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
       <div className="modal-bg border border-border rounded-xl shadow-2xl w-full max-w-lg max-h-[95vh] flex flex-col animate-in fade-in zoom-in-95">
          <div className="flex items-center justify-between p-6 border-b border-border flex-shrink-0">
            <h2 className="text-2xl font-semibold text-foreground">Sign Up</h2>
            <button
              onClick={() => {
                onSignUpClose()
                setShowSignUpForm(false)
              }}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <form onSubmit={handleSignUpSubmit} className="p-6 space-y-4 overflow-y-auto flex-1 [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Full Name</label>
              <Input
                placeholder="Enter your full name"
                value={signUpFullName}
                onChange={(e) => setSignUpFullName(e.target.value)}
                required
                className="dark:bg-input dark:text-white dark:placeholder-muted-foreground bg-background text-foreground"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Username</label>
              <div className="flex gap-2">
                <Input
                  placeholder="Choose a username"
                  value={signUpUsername}
                  onChange={(e) => {
                    const value = e.target.value.toLowerCase()
                    if (/^[a-z0-9]*$/.test(value)) {
                      setSignUpUsername(value)
                    }
                  }}
                  required
                  className="flex-1 dark:bg-input dark:text-white dark:placeholder-muted-foreground bg-background text-foreground"
                />
                {checkingUsername && <div className="px-3 py-2 text-sm text-foreground">Checking...</div>}
                {!checkingUsername && usernameAvailable === true && (
                  <div className="px-3 py-2 text-green-600 flex items-center gap-1">
                    <Check className="w-4 h-4" /> Available
                  </div>
                )}
                {!checkingUsername && usernameAvailable === false && (
                  <div className="px-3 py-2 text-red-600 text-sm">Not available</div>
                )}
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Email</label>
              <Input
                type="email"
                placeholder="Enter your email"
                value={signUpEmail}
                onChange={(e) => setSignUpEmail(e.target.value)}
                required
                className="dark:bg-input dark:text-white dark:placeholder-muted-foreground bg-background text-foreground"
              />
            </div>

            <div className="space-y-3">
              <label className="text-sm font-medium text-foreground">Password</label>
              <PasswordInput
                placeholder="Create a password"
                value={signUpPassword}
                onChange={(e) => setSignUpPassword(e.target.value)}
                required
                className="dark:bg-input dark:text-white dark:placeholder-muted-foreground bg-background text-foreground"
              />
              
              <div className="space-y-2 p-3 bg-muted/50 rounded-lg">
                <p className="text-xs font-medium text-foreground">Password requirements:</p>
                <div className="space-y-1">
                  <div className={`flex items-center gap-2 text-xs ${passwordPolicy.length ? 'text-green-600' : 'text-muted-foreground'}`}>
                    <div className={`w-3 h-3 rounded-full ${passwordPolicy.length ? 'bg-green-600' : 'bg-muted'}`} />
                    At least 9 characters
                  </div>
                  <div className={`flex items-center gap-2 text-xs ${passwordPolicy.uppercase ? 'text-green-600' : 'text-muted-foreground'}`}>
                    <div className={`w-3 h-3 rounded-full ${passwordPolicy.uppercase ? 'bg-green-600' : 'bg-muted'}`} />
                    Uppercase (A-Z)
                  </div>
                  <div className={`flex items-center gap-2 text-xs ${passwordPolicy.lowercase ? 'text-green-600' : 'text-muted-foreground'}`}>
                    <div className={`w-3 h-3 rounded-full ${passwordPolicy.lowercase ? 'bg-green-600' : 'bg-muted'}`} />
                    Lowercase (a-z)
                  </div>
                  <div className={`flex items-center gap-2 text-xs ${passwordPolicy.number ? 'text-green-600' : 'text-muted-foreground'}`}>
                    <div className={`w-3 h-3 rounded-full ${passwordPolicy.number ? 'bg-green-600' : 'bg-muted'}`} />
                    Number (0-9)
                  </div>
                  <div className={`flex items-center gap-2 text-xs ${passwordPolicy.special ? 'text-green-600' : 'text-muted-foreground'}`}>
                    <div className={`w-3 h-3 rounded-full ${passwordPolicy.special ? 'bg-green-600' : 'bg-muted'}`} />
                    Special character (!@#$%^&* etc)
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Confirm Password</label>
              <PasswordInput
                placeholder="Confirm your password"
                value={signUpConfirmPassword}
                onChange={(e) => setSignUpConfirmPassword(e.target.value)}
                required
                className="dark:bg-input dark:text-white dark:placeholder-muted-foreground bg-background text-foreground"
              />
            </div>

            <Button type="submit" className="w-full" size="lg" disabled={isSigningUp}>
              {isSigningUp ? "Creating account..." : "Sign Up"}
            </Button>

            <div className="text-center text-sm">
              <span className="text-foreground">Already have an account? </span>
              <button
                type="button"
                onClick={() => {
                  setShowSignUpForm(false)
                  onSignUpClose()
                  onOpenSignIn()
                }}
                className="text-primary hover:underline font-medium"
              >
                Sign in
              </button>
            </div>
          </form>
        </div>
      </div>
    )
  }

  return null
}
