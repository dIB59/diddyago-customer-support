import config from '../config';
import { ApiClientInterface } from './types';

export class ApiClientRest implements ApiClientInterface {

    private async request<T>(
        method: string,
        path: string,
        on_error: (messages: string[]) => void,
        data?: any
    ): Promise<T | null> {
        const full_url = new URL(path, config.api_base_url).toString();
        const headers: Record<string, string> = {
            'Accept': 'application/json',
        };
        const options: RequestInit = {
            method,
            headers,
            credentials: 'include' as const
        };
        if (method !== 'GET' && data) {
            headers['Content-Type'] = 'application/json';
            options.body = JSON.stringify(data);
        }
        try {
            const response = await fetch(full_url, options);
            if (response.status === 401) {
                const error_data = await response.json();
                on_error([error_data.detail || 'Not authenticated']);
                throw new Error('Not authenticated');
            }
            if (response.status === 403) {
                const error_data = await response.json();
                on_error([error_data.detail || 'Not authorized']);
                throw new Error('Not authorized');
            }
            if (response.status === 204) {
                return null;
            }
            if (!response.ok) {
                const error_data = await response.json();
                on_error([error_data.detail || `HTTP error! status: ${response.status}`]);
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            throw error;
        }
    }

    public get<T>(path: string, on_error: (messages: string[]) => void): Promise<T> {
        return this.request<T>('GET', path, on_error) as Promise<T>;
    }

    public post<T>(path: string, on_error: (messages: string[]) => void, data: any): Promise<T> {
        return this.request<T>('POST', path, on_error, data) as Promise<T>;
    }

    public put<T>(path: string, on_error: (messages: string[]) => void, data: any): Promise<T> {
        return this.request<T>('PUT', path, on_error, data) as Promise<T>;
    }

    public delete<T = null>(path: string, on_error: (messages: string[]) => void): Promise<T> {
        return this.request<T>('DELETE', path, on_error) as Promise<T>;
    }
}