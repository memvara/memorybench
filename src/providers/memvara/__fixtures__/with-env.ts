// src/providers/memvara/__fixtures__/with-env.ts

/** Sets environment knobs for one call and always puts them back, so an arm switched on in
 *  one test cannot leak into the next. The provider reads every knob at call time, which is
 *  what makes this enough.
 *
 *  `withEnv` is for synchronous work. Use `withEnvAsync` when the body awaits: a synchronous
 *  `finally` around a promise restores the environment while the body is still suspended,
 *  and the assertion then runs under the wrong arm. */

type Vars = Record<string, string | undefined>

function apply(vars: Vars): Map<string, string | undefined> {
  const saved = new Map<string, string | undefined>()
  for (const [name, value] of Object.entries(vars)) {
    saved.set(name, process.env[name])
    if (value === undefined) delete process.env[name]
    else process.env[name] = value
  }
  return saved
}

function restore(saved: Map<string, string | undefined>): void {
  for (const [name, value] of saved) {
    if (value === undefined) delete process.env[name]
    else process.env[name] = value
  }
}

export function withEnv<T>(vars: Vars, fn: () => T): T {
  const saved = apply(vars)
  try {
    return fn()
  } finally {
    restore(saved)
  }
}

export async function withEnvAsync<T>(vars: Vars, fn: () => Promise<T>): Promise<T> {
  const saved = apply(vars)
  try {
    return await fn()
  } finally {
    restore(saved)
  }
}
